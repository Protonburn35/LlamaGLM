// Sanitizer mlock interceptors can be no-ops. Keep real kernel locks and residency checks in standalone harnesses.
#include <cstddef>
#include <sys/syscall.h>
#include <unistd.h>

extern "C" int __wrap_mlock(const void * address, size_t bytes) {
    return int(syscall(SYS_mlock, address, bytes));
}

extern "C" int __wrap_munlock(const void * address, size_t bytes) {
    return int(syscall(SYS_munlock, address, bytes));
}
