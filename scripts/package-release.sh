#!/usr/bin/env bash
# Package the Windows CUDA build from LlamaGLM-master/build-win-cuda into one zip for C:\LlamaGLM.
# usage: scripts/package-release.sh [version]   (default: version from scripts/cluster_config.json)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${ROOT_DIR}"
BUILD_DIR="${LLAMAGLM_BUILD_DIR:-${SRC_DIR}/build-win-cuda}"
DIST_DIR="${ROOT_DIR}/dist"
STAGE_DIR="${DIST_DIR}/LlamaGLM-win-cuda-x64"
CUDA_WIN_BIN="${CUDA_WIN_BIN:-$HOME/projects/cuda128/win/root/bin}"

VERSION="${1:-$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['version'])" "${SCRIPT_DIR}/cluster_config.json")}"
SRC_COMMIT="$(git -C "${SRC_DIR}" rev-parse --short HEAD)"
ROOT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse --short HEAD)"
ARCHIVE_NAME="LlamaGLM-v${VERSION}-win-cuda-cu12.8-sm89-sm120-x64.zip"
# engine source + launchers, docs and cross-build toolchain files (outer repo)
SOURCE_NAME="LlamaGLM-v${VERSION}-source-${SRC_COMMIT}-${ROOT_COMMIT}.zip"

if [ -n "$(git -C "${SRC_DIR}" status --porcelain --untracked-files=no)" ]; then
    echo "error: ${SRC_DIR} has uncommitted changes; commit them so the package matches a source commit" >&2
    exit 1
fi
if [ -n "$(git -C "${ROOT_DIR}" status --porcelain --untracked-files=no)" ]; then
    echo "error: ${ROOT_DIR} has uncommitted changes; commit them so the package matches a launcher commit" >&2
    exit 1
fi
if [ -e "${DIST_DIR}/${ARCHIVE_NAME}" ]; then
    echo "error: ${DIST_DIR}/${ARCHIVE_NAME} exists; bump the version instead of overwriting a release" >&2
    exit 1
fi
if ! grep -q "^GGML_AVX2:BOOL=ON" "${BUILD_DIR}/CMakeCache.txt"; then
    echo "error: ${BUILD_DIR} was configured without GGML_AVX2 (cross builds default to a scalar CPU backend)" >&2
    exit 1
fi
build_commit="$(grep -o 'LLAMA_COMMIT = "[0-9a-f]*"' "${BUILD_DIR}/common/build-info.cpp" | cut -d'"' -f2 || true)"
if [ "${build_commit}" != "${SRC_COMMIT}" ]; then
    echo "error: build-win-cuda was built from '${build_commit}', source HEAD is '${SRC_COMMIT}'; rebuild first" >&2
    exit 1
fi

echo "=== Packaging LlamaGLM v${VERSION} (source ${SRC_COMMIT}) ==="
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

for exe in llama-server llama-cli llama-bench llama-fit-params ggml-rpc-server llama-rpc-test test-backend-ops llama-gguf-split; do
    cp -v "${BUILD_DIR}/bin/${exe}.exe" "${STAGE_DIR}/"
done
cp -v "${BUILD_DIR}/bin/"*.dll "${STAGE_DIR}/"

cp -v /usr/lib/gcc/x86_64-w64-mingw32/13-posix/libgcc_s_seh-1.dll "${STAGE_DIR}/"
cp -v /usr/lib/gcc/x86_64-w64-mingw32/13-posix/libstdc++-6.dll "${STAGE_DIR}/"
cp -v /usr/x86_64-w64-mingw32/lib/libwinpthread-1.dll "${STAGE_DIR}/"

for dll in cudart64_12.dll cublas64_12.dll cublasLt64_12.dll; do
    cp -v "${CUDA_WIN_BIN}/${dll}" "${STAGE_DIR}/"
done

for f in llamaglm-common.ps1 start-stage-a.ps1 start-worker.ps1 start-stage-b.ps1 start-stage-c.ps1 smoke-test.ps1 collect-logs.ps1 cluster_config.json; do
    cp -v "${SCRIPT_DIR}/${f}" "${STAGE_DIR}/"
done
for f in INSTALL-WINDOWS.md FIRST-RUN.md README.md HANDOFF.md LIMITATIONS.md; do
    cp -v "${ROOT_DIR}/${f}" "${STAGE_DIR}/"
done
cp -rv "${ROOT_DIR}/docs" "${STAGE_DIR}/"

# the zip must not carry PowerShell parse errors
if command -v pwsh > /dev/null; then
    for f in "${STAGE_DIR}"/*.ps1; do
        pwsh -NoProfile -Command "\$e=\$null; [void][System.Management.Automation.Language.Parser]::ParseFile('$f',[ref]\$null,[ref]\$e); if (\$e.Count) { \$e | % { Write-Error ('$f ' + \$_.Extent.StartLineNumber + ': ' + \$_.Message) }; exit 1 }"
    done
else
    echo "warning: pwsh not found, PowerShell scripts were not parse-checked" >&2
fi

{
    echo "LlamaGLM v${VERSION}"
    echo "source commit: $(git -C "${SRC_DIR}" rev-parse HEAD)"
    echo "launcher commit: $(git -C "${ROOT_DIR}" rev-parse HEAD)"
    echo "upstream base: llama.cpp b11780 (1242be66f)"
    echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    (cd "${STAGE_DIR}" && find . -type f ! -name MANIFEST.txt -print0 | sort -z | xargs -0 sha256sum)
} > "${STAGE_DIR}/MANIFEST.txt"

(cd "${DIST_DIR}" && zip -qr "${ARCHIVE_NAME}" "LlamaGLM-win-cuda-x64")
SRC_TMP="$(mktemp -d)"
SRC_TOP="LlamaGLM-v${VERSION}-source"
git -C "${ROOT_DIR}" archive --format=tar --prefix="${SRC_TOP}/" HEAD | tar -x -C "${SRC_TMP}"
(cd "${SRC_TMP}" && zip -qr "${DIST_DIR}/${SOURCE_NAME}" "${SRC_TOP}")
rm -rf "${SRC_TMP}"
(cd "${DIST_DIR}" && sha256sum "${ARCHIVE_NAME}" "${SOURCE_NAME}" > "LlamaGLM-v${VERSION}.SHA256SUMS")

echo "=== Package complete ==="
cat "${DIST_DIR}/LlamaGLM-v${VERSION}.SHA256SUMS"
