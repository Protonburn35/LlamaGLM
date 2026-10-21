#!/usr/bin/env bash
set -euo pipefail
if [[ "$(uname -s)" != Linux ]]; then
    echo 'This standalone sanitizer harness requires Linux.' >&2
    exit 2
fi
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_build="${1:-}"
if [[ -z "$task_build" ]]; then
    task_build="$(mktemp -d "${TMPDIR:-/tmp}/glm-storage-tsan.XXXXXX")"
fi
mkdir -p -- "$task_build"
task_cxx="${CXX:-g++}"
"$task_cxx" -std=c++17 -O1 -g -fsanitize=thread -pthread -I "$task_root/src/glm-stage" \
    "$task_root/src/glm-stage/stage-storage.cpp" \
    "$task_root/src/glm-stage/stage-expert-buffer.cpp" \
    "$task_root/src/glm-stage/stage-expert-cache.cpp" \
    "$task_root/tests/test-glm-storage.cpp" "$task_root/tests/glm-storage-tsan-locks.cpp" \
    -Wl,--wrap=mlock -Wl,--wrap=munlock -o "$task_build/test-glm-storage-tsan"
exec "$task_build/test-glm-storage-tsan"
