#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <vector>

namespace glm_stage {

constexpr uint64_t max_expert_read_bytes = 64ull << 20;

struct file_read_stats {
    uint64_t requests;
    uint64_t system_reads;
    uint64_t bytes;
};

// Read-only regular file, explicit offsets, no shared file-position cursor or model mmap.
// Windows requires a resolved local fixed volume and denies concurrent write/delete handles.
// Reads are synchronous; cancellation/deadlines and bounded async queues are not supplied here.
// Join readers before moving/destroying. A failed read permanently invalidates this object.
class expert_file {
public:
    explicit expert_file(const std::filesystem::path & path);
    ~expert_file();
    expert_file(expert_file &&) noexcept;
    expert_file & operator=(expert_file &&) noexcept;
    expert_file(const expert_file &) = delete;
    expert_file & operator=(const expert_file &) = delete;

    uint64_t size() const;
    void read_exact(uint64_t offset, uint8_t * output, size_t bytes) const;
    file_read_stats stats() const;

private:
    struct impl;
    std::unique_ptr<impl> state;
};

struct expert_address {
    uint32_t file;
    uint64_t offset;
    uint64_t bytes;
    bool operator<(const expert_address & other) const;
};

using expert_lease = std::shared_ptr<const std::vector<uint8_t>>;

// Single-threaded preload/seal, immutable concurrent lookups afterwards. No file handle/read callback.
// This stores exact raw expert bytes; it does not pin host pages or prove Windows/GPU residency.
class memory_experts {
public:
    memory_experts(uint64_t byte_budget, size_t entry_budget);
    void preload(expert_address address, const uint8_t * data, size_t bytes);
    void seal(const std::vector<expert_address> & complete_owned_catalog);
    expert_lease acquire(expert_address address) const;
    uint64_t payload_bytes() const { return used; }

private:
    uint64_t budget;
    uint64_t used = 0;
    size_t entry_budget;
    bool sealed = false;
    std::map<expert_address, expert_lease> entries;
};

} // namespace glm_stage
