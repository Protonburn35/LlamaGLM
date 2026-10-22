# Cross-compile llama.cpp for Windows x64 from Linux using MinGW-w64.
# posix-thread mingw variants + static winpthread/libgcc/libstdc++ so the
# resulting DLLs/EXEs need no mingw runtime DLLs.
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_C_COMPILER   /usr/bin/x86_64-w64-mingw32-gcc-posix)
set(CMAKE_CXX_COMPILER /usr/bin/x86_64-w64-mingw32-g++-posix)
set(CMAKE_RC_COMPILER  /usr/bin/x86_64-w64-mingw32-windres)

set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# NOTE: do NOT statically link libgcc/libstdc++/winpthread here: mingw-w64
# GCC 13 hits "multiple definition of _Unwind_Resume" when static libgcc_eh
# is mixed with DLL import libraries. Instead we ship
#   libgcc_s_seh-1.dll, libstdc++-6.dll, libwinpthread-1.dll
# alongside the build (they live in /usr/lib/gcc/x86_64-w64-mingw32/13*/
# and /usr/x86_64-w64-mingw32/bin/).
