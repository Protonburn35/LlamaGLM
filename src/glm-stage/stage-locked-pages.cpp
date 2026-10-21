#include "stage-locked-pages.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <cerrno>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace glm_stage {
namespace {
std::atomic<uint64_t> locked_payload_bytes{0}, locked_payload_buffers{0};

size_t page_size() {
#ifdef _WIN32
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    const auto bytes = size_t(info.dwPageSize);
#else
    const auto queried = sysconf(_SC_PAGESIZE);
    if (queried <= 0) { throw std::runtime_error("cannot query native host page size"); }
    const auto bytes = size_t(queried);
#endif
    if (bytes == 0 || (bytes & (bytes - 1)) != 0 || bytes > (1u << 20)) { throw std::runtime_error("unsupported host page size"); }
    return bytes;
}
}

struct locked_pages::impl {
    uint8_t * data = nullptr;
    size_t bytes = 0, allocated = 0;
    bool locked = false, counted = false;
    ~impl() {
        if (!data) { return; }
#ifdef _WIN32
        if (locked) { VirtualUnlock(data, allocated); }
        VirtualFree(data, 0, MEM_RELEASE);
#else
        if (locked) { munlock(data, allocated); }
        munmap(data, allocated);
#endif
        if (counted) { locked_payload_bytes -= allocated; --locked_payload_buffers; }
    }
};

uint64_t locked_pages::required_bytes(size_t bytes) {
    if (bytes == 0 || bytes > max_locked_buffer_bytes) { throw std::invalid_argument("invalid locked host allocation size"); }
    const size_t page = page_size();
    if (bytes > std::numeric_limits<size_t>::max() - (page - 1)) { throw std::overflow_error("host page rounding overflow"); }
    return (bytes + page - 1) & ~(page - 1);
}

locked_pages::locked_pages(size_t bytes, uint64_t page_budget) : state(std::make_unique<impl>()) {
    state->bytes = bytes;
    state->allocated = size_t(required_bytes(bytes));
    if (page_budget == 0 || page_budget > max_locked_buffer_bytes || state->allocated > page_budget) {
        throw std::invalid_argument("locked host allocation exceeds its reserved page budget");
    }
#ifdef _WIN32
    state->data = static_cast<uint8_t *>(VirtualAlloc(nullptr, state->allocated, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!state->data) { throw std::system_error(int(GetLastError()), std::system_category(), "host buffer VirtualAlloc failed"); }
    if (!VirtualLock(state->data, state->allocated)) {
        throw std::system_error(int(GetLastError()), std::system_category(), "host buffer VirtualLock failed; no pageable fallback");
    }
#else
    void * allocation = mmap(nullptr, state->allocated, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (allocation == MAP_FAILED) { throw std::system_error(errno, std::system_category(), "anonymous host allocation failed"); }
    state->data = static_cast<uint8_t *>(allocation);
    if (mlock(state->data, state->allocated) != 0) {
        throw std::system_error(errno, std::system_category(), "host buffer mlock failed; no pageable fallback");
    }
#endif
    state->locked = true;
    verify_locked();
    locked_payload_bytes += state->allocated;
    ++locked_payload_buffers;
    state->counted = true;
}

locked_pages::~locked_pages() = default;
uint8_t * locked_pages::data() { return state->data; }
const uint8_t * locked_pages::data() const { return state->data; }
size_t locked_pages::size() const { return state->bytes; }
uint64_t locked_pages::locked_bytes() const { return state->locked ? state->allocated : 0; }
locked_buffer_stats locked_pages::locked_stats() { return {locked_payload_bytes.load(), locked_payload_buffers.load()}; }

void locked_pages::verify_locked() const {
    if (!state->locked || !state->data) { throw std::runtime_error("host buffer has no native page lock"); }
    const size_t page = page_size();
    constexpr size_t batch = 256;
#ifdef _WIN32
    std::array<PSAPI_WORKING_SET_EX_INFORMATION, batch> info{};
    for (size_t offset = 0; offset < state->allocated;) {
        size_t count = 0;
        for (; count < batch && offset < state->allocated; ++count, offset += page) { info[count].VirtualAddress = state->data + offset; }
        if (!QueryWorkingSetEx(GetCurrentProcess(), info.data(), DWORD(count * sizeof(info[0])))) {
            throw std::system_error(int(GetLastError()), std::system_category(), "host working-set query failed");
        }
        for (size_t i = 0; i < count; ++i) {
            if (!info[i].VirtualAttributes.Valid || !info[i].VirtualAttributes.Locked) {
                throw std::runtime_error("host buffer page is not locked in the Windows working set");
            }
        }
    }
#else
    std::array<unsigned char, batch> resident{};
    for (size_t offset = 0; offset < state->allocated;) {
        const size_t bytes = std::min(batch * page, state->allocated - offset);
        if (mincore(state->data + offset, bytes, resident.data()) != 0) {
            throw std::system_error(errno, std::system_category(), "host resident-page query failed");
        }
        for (size_t i = 0; i < bytes / page; ++i) {
            if (!(resident[i] & 1)) { throw std::runtime_error("mlocked host page is not resident"); }
        }
        offset += bytes;
    }
#endif
}

} // namespace glm_stage
