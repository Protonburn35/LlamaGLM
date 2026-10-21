#!/usr/bin/env bash
set -euo pipefail
if [[ "$(uname -s)" != Linux ]]; then
    echo 'This standalone sanitizer harness requires Linux.' >&2
    exit 2
fi
task_mode="${1:-address}"
case "$task_mode" in
    address) task_sanitizers=address,undefined ;;
    thread) task_sanitizers=thread ;;
    *) echo 'Choose address or thread.' >&2; exit 2 ;;
esac
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_build="${2:-}"
if [[ -z "$task_build" ]]; then
    task_build="$(mktemp -d "${TMPDIR:-/tmp}/glm-credentials-sanitizer.XXXXXX")"
fi
mkdir -p -- "$task_build"
task_cxx="${CXX:-g++}"
"$task_cxx" -std=c++17 -O1 -g -fsanitize="$task_sanitizers" -fno-sanitize-recover=all -fno-omit-frame-pointer -pthread \
    -I "$task_root/tools/glm-rpc" -I "$task_root/src/glm-stage" \
    "$task_root/tools/glm-rpc/stage-crypto.cpp" "$task_root/tools/glm-rpc/stage-wire.cpp" \
    "$task_root/tools/glm-rpc/stage-credential.cpp" "$task_root/src/glm-stage/stage-storage.cpp" \
    "$task_root/src/glm-stage/stage-locked-pages.cpp" "$task_root/src/glm-stage/stage-expert-buffer.cpp" \
    "$task_root/src/glm-stage/stage-expert-cache.cpp" "$task_root/tests/test-glm-credentials.cpp" \
    "$task_root/tests/glm-storage-tsan-locks.cpp" -Wl,--wrap=mlock -Wl,--wrap=munlock \
    -lcrypto -o "$task_build/test-glm-credentials-$task_mode"
exec python3 "$task_root/tests/test-glm-credentials.py" --executable "$task_build/test-glm-credentials-$task_mode"
