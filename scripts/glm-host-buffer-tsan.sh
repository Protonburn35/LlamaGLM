#!/usr/bin/env bash
set -euo pipefail
if [[ "$(uname -s)" != Linux ]]; then
    echo 'This standalone sanitizer harness requires Linux.' >&2
    exit 2
fi
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_ggml="${1:?Pass a shared CPU GGML build with TSan and OpenMP disabled.}"
task_output="${2:-}"
if [[ -z "$task_output" ]]; then
    task_output="$(mktemp -d "${TMPDIR:-/tmp}/glm-host-buffer-tsan.XXXXXX")"
fi
for task_library in ggml ggml-base ggml-cpu; do
    if [[ ! -f "$task_ggml/bin/lib$task_library.so" ]]; then
        echo 'The sanitizer harness requires a shared CPU GGML build.' >&2
        exit 2
    fi
done
if [[ ! -f "$task_ggml/CMakeCache.txt" ]] ||
        ! rg -q '^GGML_OPENMP:BOOL=OFF$' "$task_ggml/CMakeCache.txt" ||
        ! rg -q '^CMAKE_C_FLAGS:STRING=.*-fsanitize=thread' "$task_ggml/CMakeCache.txt" ||
        ! rg -q '^CMAKE_CXX_FLAGS:STRING=.*-fsanitize=thread' "$task_ggml/CMakeCache.txt"; then
    echo 'Build CPU GGML with matching thread instrumentation and GGML_OPENMP=OFF before running this harness.' >&2
    exit 2
fi
mkdir -p -- "$task_output"
task_cxx="${CXX:-g++}"
"$task_cxx" -std=c++17 -O1 -g -fsanitize=thread -pthread \
    -I "$task_root/src/glm-stage" -I "$task_root/ggml/include" -I "$task_root/ggml/src" \
    "$task_root/src/glm-stage/stage-locked-pages.cpp" "$task_root/src/glm-stage/stage-host-buffer.cpp" \
    "$task_root/tests/test-glm-host-buffer.cpp" "$task_root/tests/glm-storage-tsan-locks.cpp" \
    -Wl,--wrap=mlock -Wl,--wrap=munlock -L "$task_ggml/bin" -Wl,-rpath,"$task_ggml/bin" \
    -lggml -lggml-cpu -lggml-base -o "$task_output/test-glm-host-buffer-tsan"
exec "$task_output/test-glm-host-buffer-tsan"
