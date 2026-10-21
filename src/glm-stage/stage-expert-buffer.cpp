#include "stage-storage.h"

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
    if (queried <= 0) { throw std::runtime_error("cannot query native expert page size"); }
    const auto bytes = size_t(queried);
#endif
    if (bytes == 0 || (bytes & (bytes - 1)) != 0 || bytes > (1u << 20)) { throw std::runtime_error("unsupported expert page size"); }
    return bytes;
}
}

struct expert_buffer::impl {
    uint8_t * data = nullptr;
    size_t bytes = 0, allocated = 0;
    bool native = false, locked = false, counted = false;
    ~impl() {
        if (!data) { return; }
        if (native) {
#ifdef _WIN32
            if (locked) { VirtualUnlock(data, allocated); }
            VirtualFree(data, 0, MEM_RELEASE);
#else
            if (locked) { munlock(data, allocated); }
            munmap(data, allocated);
#endif
        } else { delete[] data; }
        if (counted) { locked_payload_bytes -= allocated; --locked_payload_buffers; }
    }
};

uint64_t expert_buffer::required_locked_bytes(size_t bytes) {
    if (bytes == 0 || bytes > max_expert_read_bytes) { throw std::invalid_argument("invalid expert payload allocation size"); }
    const size_t page = page_size();
    if (bytes > std::numeric_limits<size_t>::max() - (page - 1)) { throw std::overflow_error("expert page rounding overflow"); }
    return (bytes + page - 1) & ~(page - 1);
}

expert_buffer::expert_buffer(size_t bytes, bool lock_pages) : state(std::make_unique<impl>()) {
    if (bytes == 0 || bytes > max_expert_read_bytes) { throw std::invalid_argument("invalid expert payload allocation size"); }
    state->bytes = bytes;
    if (!lock_pages) { state->data = new uint8_t[bytes](); return; }
    state->allocated = size_t(required_locked_bytes(bytes));
    state->native = true;
#ifdef _WIN32
    state->data = static_cast<uint8_t *>(VirtualAlloc(nullptr, state->allocated, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!state->data) { throw std::system_error(int(GetLastError()), std::system_category(), "expert VirtualAlloc failed"); }
    if (!VirtualLock(state->data, state->allocated)) {
        throw std::system_error(int(GetLastError()), std::system_category(), "expert VirtualLock failed; no pageable fallback");
    }
#else
    void * allocation = mmap(nullptr, state->allocated, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (allocation == MAP_FAILED) { throw std::system_error(errno, std::system_category(), "anonymous expert allocation failed"); }
    state->data = static_cast<uint8_t *>(allocation);
    if (mlock(state->data, state->allocated) != 0) {
        throw std::system_error(errno, std::system_category(), "expert mlock failed; no pageable fallback");
    }
#endif
    state->locked = true;
    verify_locked();
    locked_payload_bytes += state->allocated;
    ++locked_payload_buffers;
    state->counted = true;
}

expert_buffer::~expert_buffer() = default;
uint8_t * expert_buffer::data() { return state->data; }
const uint8_t * expert_buffer::data() const { return state->data; }
size_t expert_buffer::size() const { return state->bytes; }
uint64_t expert_buffer::locked_bytes() const { return state->locked ? state->allocated : 0; }
locked_buffer_stats expert_buffer::locked_stats() { return {locked_payload_bytes.load(), locked_payload_buffers.load()}; }

void expert_buffer::verify_locked() const {
    if (!state->native || !state->locked || !state->data) { throw std::runtime_error("expert payload has no native page lock"); }
    const size_t page = page_size();
    constexpr size_t batch = 256;
#ifdef _WIN32
    std::array<PSAPI_WORKING_SET_EX_INFORMATION, batch> info{};
    for (size_t offset = 0; offset < state->allocated;) {
        size_t count = 0;
        for (; count < batch && offset < state->allocated; ++count, offset += page) { info[count].VirtualAddress = state->data + offset; }
        if (!QueryWorkingSetEx(GetCurrentProcess(), info.data(), DWORD(count * sizeof(info[0])))) {
            throw std::system_error(int(GetLastError()), std::system_category(), "expert working-set query failed");
        }
        for (size_t i = 0; i < count; ++i) {
            if (!info[i].VirtualAttributes.Valid || !info[i].VirtualAttributes.Locked) {
                throw std::runtime_error("expert payload page is not locked in the Windows working set");
            }
        }
    }
#else
    std::array<unsigned char, batch> resident{};
    for (size_t offset = 0; offset < state->allocated;) {
        const size_t bytes = std::min(batch * page, state->allocated - offset);
        if (mincore(state->data + offset, bytes, resident.data()) != 0) {
            throw std::system_error(errno, std::system_category(), "expert resident-page query failed");
        }
        for (size_t i = 0; i < bytes / page; ++i) {
            if (!(resident[i] & 1)) { throw std::runtime_error("mlocked expert page is not resident"); }
        }
        offset += bytes;
    }
#endif
}

} // namespace glm_stage
