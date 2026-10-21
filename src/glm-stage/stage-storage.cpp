#include "stage-storage.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <set>
#include <stdexcept>
#include <system_error>
#include <tuple>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace glm_stage {

struct expert_file::impl {
    uint64_t bytes = 0;
    std::atomic<bool> failed{false};
    std::atomic<uint64_t> requests{0}, system_reads{0}, transferred{0};
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    ~impl() { if (file != INVALID_HANDLE_VALUE) { CloseHandle(file); } }
#else
    int file = -1;
    struct stat original{};
    ~impl() { if (file >= 0) { close(file); } }
    void check_identity() const {
        struct stat current{};
        if (fstat(file, &current) != 0) { throw std::system_error(errno, std::system_category(), "GLM file stat failed"); }
#ifdef __APPLE__
        const auto mtime = current.st_mtimespec, old_mtime = original.st_mtimespec;
        const auto ctime = current.st_ctimespec, old_ctime = original.st_ctimespec;
#else
        const auto mtime = current.st_mtim, old_mtime = original.st_mtim;
        const auto ctime = current.st_ctim, old_ctime = original.st_ctim;
#endif
        if (current.st_dev != original.st_dev || current.st_ino != original.st_ino || current.st_size != original.st_size ||
                mtime.tv_sec != old_mtime.tv_sec || mtime.tv_nsec != old_mtime.tv_nsec ||
                ctime.tv_sec != old_ctime.tv_sec || ctime.tv_nsec != old_ctime.tv_nsec) {
            throw std::runtime_error("GLM expert file changed after opening");
        }
    }
#endif
};

expert_file::expert_file(const std::filesystem::path & path) : state(std::make_unique<impl>()) {
#ifdef _WIN32
    state->file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (state->file == INVALID_HANDLE_VALUE) { throw std::system_error(int(GetLastError()), std::system_category(), "GLM file open failed"); }
    FILE_STANDARD_INFO info{};
    if (GetFileType(state->file) != FILE_TYPE_DISK || !GetFileInformationByHandleEx(state->file, FileStandardInfo, &info, sizeof(info)) ||
            info.Directory || info.EndOfFile.QuadPart < 0) { throw std::runtime_error("GLM expert source must be a regular disk file"); }
    std::vector<wchar_t> resolved(32768), volume(32768);
    const DWORD length = GetFinalPathNameByHandleW(state->file, resolved.data(), DWORD(resolved.size()), FILE_NAME_NORMALIZED);
    if (length == 0) { throw std::system_error(int(GetLastError()), std::system_category(), "GLM final file path query failed"); }
    if (length >= resolved.size()) { throw std::invalid_argument("GLM final file path exceeds limit"); }
    if (!GetVolumePathNameW(resolved.data(), volume.data(), DWORD(volume.size()))) {
        throw std::system_error(int(GetLastError()), std::system_category(), "GLM local volume query failed");
    }
    if (GetDriveTypeW(volume.data()) != DRIVE_FIXED) { throw std::runtime_error("GLM expert source must resolve to a local fixed volume"); }
    state->bytes = uint64_t(info.EndOfFile.QuadPart);
#else
    static_assert(sizeof(off_t) >= 8, "GLM expert files require 64-bit positional I/O");
    state->file = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (state->file < 0) { throw std::system_error(errno, std::system_category(), "GLM file open failed"); }
    if (fstat(state->file, &state->original) != 0 || !S_ISREG(state->original.st_mode) || state->original.st_size < 0) {
        throw std::runtime_error("GLM expert source must be a regular disk file");
    }
    state->bytes = uint64_t(state->original.st_size);
#endif
    if (state->bytes > (1ull << 44)) { throw std::invalid_argument("GLM expert file exceeds size limit"); }
}

expert_file::~expert_file() = default;
expert_file::expert_file(expert_file &&) noexcept = default;
expert_file & expert_file::operator=(expert_file &&) noexcept = default;

uint64_t expert_file::size() const {
    if (!state) { throw std::runtime_error("GLM expert file was moved"); }
    return state->bytes;
}

file_read_stats expert_file::stats() const {
    if (!state) { throw std::runtime_error("GLM expert file was moved"); }
    return {state->requests.load(), state->system_reads.load(), state->transferred.load()};
}

void expert_file::read_exact(uint64_t offset, uint8_t * output, size_t bytes) const {
    if (!state || state->failed.load()) { throw std::runtime_error("GLM expert file is invalid; reopen and revalidate"); }
    try {
        if (!output || bytes == 0 || bytes > max_expert_read_bytes || offset > state->bytes || bytes > state->bytes - offset) {
            throw std::invalid_argument("GLM expert read exceeds validated file span or buffer limit");
        }
        ++state->requests;
#ifdef _WIN32
        struct event {
            HANDLE value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            ~event() { if (value) { CloseHandle(value); } }
        } completion;
        if (!completion.value) { throw std::system_error(int(GetLastError()), std::system_category(), "GLM read event failed"); }
#else
        state->check_identity();
#endif
        size_t done = 0;
        while (done < bytes) {
            if (state->failed.load()) { throw std::runtime_error("GLM expert source failed during another read"); }
            const size_t request = std::min<size_t>(bytes - done, 1u << 20);
            ++state->system_reads;
#ifdef _WIN32
            OVERLAPPED operation{};
            operation.Offset = DWORD((offset + done) & 0xffffffffu);
            operation.OffsetHigh = DWORD((offset + done) >> 32);
            operation.hEvent = completion.value;
            if (!ResetEvent(completion.value)) { throw std::system_error(int(GetLastError()), std::system_category(), "GLM read event reset failed"); }
            if (!ReadFile(state->file, output + done, DWORD(request), nullptr, &operation) && GetLastError() != ERROR_IO_PENDING) {
                throw std::system_error(int(GetLastError()), std::system_category(), "GLM positional read failed");
            }
            DWORD transferred = 0;
            // Keep OVERLAPPED, event and destination alive until this exact I/O completes.
            if (!GetOverlappedResult(state->file, &operation, &transferred, TRUE)) {
                throw std::system_error(int(GetLastError()), std::system_category(), "GLM positional read completion failed");
            }
#else
            const ssize_t result = pread(state->file, output + done, request, off_t(offset + done));
            if (result < 0 && errno == EINTR) { continue; }
            if (result < 0) { throw std::system_error(errno, std::system_category(), "GLM positional read failed"); }
            const size_t transferred = size_t(result);
#endif
            if (transferred == 0 || transferred > request) { throw std::runtime_error("GLM expert source returned a short or invalid read"); }
            state->transferred += transferred;
            done += transferred;
        }
#ifndef _WIN32
        state->check_identity();
#endif
        if (state->failed.load()) { throw std::runtime_error("GLM expert source failed during another read"); }
    } catch (...) { state->failed.store(true); throw; }
}

bool expert_address::operator<(const expert_address & other) const {
    return std::tie(file, offset, bytes) < std::tie(other.file, other.offset, other.bytes);
}

memory_experts::memory_experts(uint64_t byte_budget, size_t entry_budget, uint64_t page_lock_budget) :
    budget(byte_budget), lock_budget(page_lock_budget), entry_budget(entry_budget) {
    if (budget == 0 || budget > (1ull << 40) || entry_budget == 0 || entry_budget > 100000 || lock_budget > (1ull << 40)) {
        throw std::invalid_argument("invalid GLM memory expert store budget");
    }
}

void memory_experts::preload(expert_address address, const uint8_t * data, size_t bytes) {
    if (sealed || !data || address.file >= 100 || bytes == 0 || address.bytes != bytes || bytes > max_expert_read_bytes || bytes > budget - used ||
            address.offset > (1ull << 44) - bytes || entries.size() >= entry_budget || entries.count(address)) {
        throw std::invalid_argument("GLM expert preload is sealed, duplicated or exceeds local bounds");
    }
    const uint64_t pages = lock_budget ? expert_buffer::required_locked_bytes(bytes) : 0;
    if (pages > lock_budget - locked_used) { throw std::invalid_argument("expert payload exceeds the separate page-lock budget"); }
    auto values = std::make_shared<expert_buffer>(bytes, lock_budget != 0);
    std::memcpy(values->data(), data, bytes);
    entries.emplace(address, std::move(values));
    used += bytes;
    locked_used += pages;
}

void memory_experts::seal(const std::vector<expert_address> & complete_owned_catalog) {
    if (sealed || complete_owned_catalog.empty() || complete_owned_catalog.size() != entries.size()) {
        throw std::invalid_argument("GLM resident catalog is incomplete or already sealed");
    }
    std::set<expert_address> unique;
    for (const auto & address : complete_owned_catalog) {
        if (!unique.insert(address).second || !entries.count(address)) { throw std::invalid_argument("GLM resident catalog has missing/duplicate experts"); }
    }
    sealed = true;
}

expert_lease memory_experts::acquire(expert_address address) const {
    if (!sealed) { throw std::runtime_error("GLM memory expert store is not sealed"); }
    const auto found = entries.find(address);
    if (found == entries.end()) { throw std::runtime_error("GLM selected expert is absent; no disk fallback exists in the memory store"); }
    return found->second;
}

void memory_experts::verify_payload_locks() const {
    if (!payloads_locked()) { throw std::runtime_error("expert store has no sealed locked-payload policy"); }
    for (const auto & entry : entries) { entry.second->verify_locked(); }
}

} // namespace glm_stage
