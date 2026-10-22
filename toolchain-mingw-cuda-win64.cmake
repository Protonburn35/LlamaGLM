# Cross-compile llama.cpp WITH CUDA for Windows x64 from Linux.
# - C/C++  : MinGW-w64 (same runtime policy as the CPU build: ship mingw DLLs)
# - .cu    : clang-20 (CUDA frontend) targeting x86_64-w64-mingw32;
#            device code assembled by Linux ptxas from the CUDA 12.8 redist
#            nvcc package (OS-independent cubins), fatbins embedded into COFF.
# - CUDA   : redist packages assembled in ~/projects/cuda128:
#              host/root = linux tools+headers, win/root = windows includes,
#              DLLs and MSVC import libs; cross/ merges them for clang.
if (NOT DEFINED ENV{CUDA_CROSS})
  set(CUDA_CROSS "$ENV{HOME}/projects/cuda128/cross" CACHE PATH "cuda cross prefix")
else()
  set(CUDA_CROSS "$ENV{CUDA_CROSS}" CACHE PATH "cuda cross prefix")
endif()

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_C_COMPILER   /usr/bin/x86_64-w64-mingw32-gcc-posix)
set(CMAKE_CXX_COMPILER /usr/bin/x86_64-w64-mingw32-g++-posix)
set(CMAKE_RC_COMPILER  /usr/bin/x86_64-w64-mingw32-windres)

set(CMAKE_CUDA_COMPILER "${CMAKE_CURRENT_LIST_DIR}/cuda-clang-wrap.sh" CACHE FILEPATH "cuda compiler: clang compile + mingw g++ link wrapper")
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/x86_64-w64-mingw32-g++-posix CACHE FILEPATH "")
set(CMAKE_CUDA_FLAGS_INIT "--target=x86_64-w64-mingw32 -isystem /usr/lib/gcc/x86_64-w64-mingw32/13-posix/include/c++ -isystem /usr/lib/gcc/x86_64-w64-mingw32/13-posix/include/c++/x86_64-w64-mingw32 --cuda-path=${CUDA_CROSS} -DCLANG_CUDA_CMATH_SKIP")
# CMake's CMakeDetermineCUDACompiler refuses Clang+CUDA when
# CMAKE_SYSTEM_NAME=Windows (even though our HOST is Linux and this is a
# plain cross-compile). Pre-seed the compiler id to skip that block.
set(CMAKE_CUDA_COMPILER_ID "Clang" CACHE STRING "")
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_COMPILER_VERSION "20.1.2" CACHE STRING "")
set(CMAKE_CUDA_SIMULATE_ID "" CACHE STRING "")
set(CMAKE_CUDA_SIMULATE_VERSION "" CACHE STRING "")
set(CMAKE_CUDA_ARCHITECTURES "89;120" CACHE STRING "")
set(CMAKE_CUDA_DEVICE_LINKER "${CUDA_CROSS}/bin/nvlink" CACHE FILEPATH "")
set(CMAKE_CUDA_FATBINARY "${CUDA_CROSS}/bin/fatbinary" CACHE FILEPATH "")
set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "${CUDA_CROSS}" CACHE PATH "")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "${CUDA_CROSS}" CACHE PATH "")
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "${CUDA_CROSS}/include" CACHE PATH "")
# normally computed by the ABI/ID detection tries we skipped:
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "GNU" CACHE STRING "")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "14" CACHE STRING "")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON" CACHE STRING "")
macro(cmake_record_cuda_compile_features)
endmacro()
# no CMake-managed cudart/cudadevrt linkage; ggml links the win import libs
# (cudart.lib etc via CUDA:: imported targets) itself
set(CMAKE_CUDA_RUNTIME_LIBRARY "None" CACHE STRING "")

set(CUDAToolkit_ROOT "${CUDA_CROSS}" CACHE PATH "cuda toolkit root (cross prefix)")

# same trick as llama.cpp's cmake/arm64-windows-msvc-cuda.cmake: seed
# FindCUDAToolkit with explicit paths (import libs are the Windows x64 ones)
set(CUDA_CUDART               "${CUDA_CROSS}/lib" CACHE PATH "")
set(CUDA_cudart_LIBRARY       "${CUDA_CROSS}/lib/cudart.lib" CACHE FILEPATH "")
set(CUDA_cudart_static_LIBRARY "${CUDA_CROSS}/lib/cudart_static.lib" CACHE FILEPATH "")
set(CUDA_cublas_LIBRARY       "${CUDA_CROSS}/lib/cublas.lib" CACHE FILEPATH "")
set(CUDA_cublasLt_LIBRARY     "${CUDA_CROSS}/lib/cublasLt.lib" CACHE FILEPATH "")
set(CUDA_curand_LIBRARY       "${CUDA_CROSS}/lib/curand.lib" CACHE FILEPATH "")
set(CUDA_cuda_LIBRARY         "${CUDA_CROSS}/lib/cuda.lib" CACHE FILEPATH "")
set(CUDA_cuda_driver_LIBRARY  "${CUDA_CROSS}/lib/cuda.lib" CACHE FILEPATH "")
set(CUDAToolkit_INCLUDE_DIRS  "${CUDA_CROSS}/include" CACHE PATH "")

set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
# NOTE: no static mingw runtimes - mingw gcc13 + DLL import libs collide on
# _Unwind_Resume; we ship libgcc/libstdc++/libwinpthread DLLs instead.
