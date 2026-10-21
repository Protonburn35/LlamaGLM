#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// Test process quota only. Production must admit its own resources explicitly.
struct glm_fixture_lock_quota {
#ifdef _WIN32
    SIZE_T minimum = 0, maximum = 0;
    DWORD flags = 0;
#endif
    glm_fixture_lock_quota() {
#ifdef _WIN32
        if (!GetProcessWorkingSetSizeEx(GetCurrentProcess(), &minimum, &maximum, &flags) ||
                !SetProcessWorkingSetSizeEx(GetCurrentProcess(), std::max<SIZE_T>(minimum, 16u << 20), std::max<SIZE_T>(maximum, 64u << 20), flags)) {
            throw std::runtime_error("cannot reserve test process working-set quota for native lock qualification");
        }
#endif
    }
    ~glm_fixture_lock_quota() {
#ifdef _WIN32
        if (!SetProcessWorkingSetSizeEx(GetCurrentProcess(), minimum, maximum, flags)) {
            std::fputs("cannot restore test process working-set quota\n", stderr);
            std::abort();
        }
#endif
    }
    glm_fixture_lock_quota(const glm_fixture_lock_quota &) = delete;
    glm_fixture_lock_quota & operator=(const glm_fixture_lock_quota &) = delete;
};
