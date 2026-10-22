#!/bin/sh
# clang for CUDA compiles, mingw g++ driver for links (mingw CRT ordering).
for a in "$@"; do
  if [ "$a" = "-c" ] || [ "$a" = "--cuda-device-only" ]; then
    exec /usr/lib/llvm-20/bin/clang++ "$@"
  fi
done
exec /usr/bin/x86_64-w64-mingw32-g++-posix "$@"
