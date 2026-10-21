#include "stage-storage.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <functional>
#include <future>
#include <random>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winioctl.h>
#endif

using namespace glm_stage;

static void require(bool condition, const char * message) { if (!condition) { throw std::runtime_error(message); } }
static void refused(const std::function<void()> & action) {
    try { action(); } catch (const std::exception &) { return; }
    throw std::runtime_error("invalid storage operation was accepted");
}

struct fixture {
    std::filesystem::path directory = std::filesystem::temp_directory_path() / ("glm storage fixture " + std::to_string(std::random_device{}()));
    std::filesystem::path file;
    std::vector<uint8_t> data = std::vector<uint8_t>((2u << 20) + 29);
    fixture() {
        require(std::filesystem::create_directory(directory), "cannot create isolated storage fixture");
        file = directory / std::filesystem::u8path(u8"expert \u03b1 bytes.bin");
        for (size_t i = 0; i < data.size(); ++i) { data[i] = uint8_t(i * 17 + i / 251); }
        std::ofstream output(file, std::ios::binary);
        output.write(reinterpret_cast<const char *>(data.data()), data.size());
        output.close();
        require(bool(output), "cannot write synthetic expert source");
    }
    ~fixture() { std::error_code error; std::filesystem::remove_all(directory, error); }
};

static void file_reads() {
    fixture f;
    expert_file file(f.file);
    require(file.size() == f.data.size(), "native file size differs");
    std::vector<uint8_t> actual(f.data.size());
    file.read_exact(0, actual.data(), actual.size());
    require(actual == f.data, "native multi-part read differs from source");
    std::array<uint8_t, 129> tail{};
    file.read_exact(file.size() - tail.size(), tail.data(), tail.size());
    require(std::equal(tail.begin(), tail.end(), f.data.end() - tail.size()), "unaligned exact tail read differs");
    const auto stats = file.stats();
    require(stats.requests == 2 && stats.system_reads >= 4 && stats.bytes == actual.size() + tail.size(), "native I/O counters differ");
    expert_file moved(std::move(file));
    refused([&] { file.read_exact(0, tail.data(), tail.size()); });
    moved.read_exact(3, tail.data(), tail.size());
    require(std::equal(tail.begin(), tail.end(), f.data.begin() + 3), "moved file lost its independent offset");
}

static void concurrent_reads() {
    fixture f;
    expert_file file(f.file);
    std::vector<std::future<void>> readers;
    for (size_t worker = 0; worker < 8; ++worker) {
        readers.push_back(std::async(std::launch::async, [&, worker] {
            std::array<uint8_t, 4093> actual{};
            for (size_t i = 0; i < 25; ++i) {
                const size_t offset = (worker * 17011 + i * 8191) % (f.data.size() - actual.size());
                file.read_exact(offset, actual.data(), actual.size());
                require(std::equal(actual.begin(), actual.end(), f.data.begin() + offset), "shared handle positional reads raced");
            }
        }));
    }
    for (auto & reader : readers) { reader.get(); }
    require(file.stats().requests == 200 && file.stats().bytes == 200 * 4093, "concurrent read counters differ");
}

static void large_offsets() {
    fixture f;
    const auto path = f.directory / "large sparse synthetic file.bin";
    { std::ofstream create(path, std::ios::binary); require(bool(create), "cannot create sparse fixture"); }
#ifdef _WIN32
    const auto handle = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    require(handle != INVALID_HANDLE_VALUE, "cannot open own sparse fixture");
    DWORD returned = 0;
    const bool sparse = DeviceIoControl(handle, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr) != 0;
    CloseHandle(handle);
    require(sparse, "fixture volume cannot mark a synthetic sparse file");
#endif
    constexpr uint64_t offset = (1ull << 32) + 37;
    const std::array<uint8_t, 9> expected{1,3,5,7,9,11,13,15,17};
    {
        std::fstream output(path, std::ios::binary | std::ios::in | std::ios::out);
        output.seekp(std::streamoff(offset));
        output.write(reinterpret_cast<const char *>(expected.data()), expected.size());
        output.close();
        require(bool(output), "cannot write synthetic high-offset marker");
    }
    expert_file file(path);
    std::array<uint8_t, 9> actual{};
    file.read_exact(offset, actual.data(), actual.size());
    require(file.size() == offset + expected.size() && actual == expected, "64-bit expert file offset was truncated");
}

static void invalid_reads() {
    fixture f;
    std::array<uint8_t, 16> data{};
    for (int failure = 0; failure < 5; ++failure) {
        expert_file file(f.file);
        refused([&] {
            if (failure == 0) { file.read_exact(file.size() - 1, data.data(), data.size()); }
            if (failure == 1) { file.read_exact(UINT64_MAX, data.data(), data.size()); }
            if (failure == 2) { file.read_exact(0, nullptr, data.size()); }
            if (failure == 3) { file.read_exact(0, data.data(), 0); }
            if (failure == 4) { file.read_exact(0, data.data(), max_expert_read_bytes + 1); }
        });
        require(file.stats().system_reads == 0, "invalid range reached native I/O");
        refused([&] { file.read_exact(0, data.data(), data.size()); });
    }
    refused([&] { expert_file directory(f.directory); });
    refused([&] { expert_file missing(f.directory / "absent.bin"); });
}

static void changed_source() {
    fixture f;
    expert_file file(f.file);
#ifdef _WIN32
    // Windows must deny a write handle while the inference read handle exists.
    const auto writer = CreateFileW(f.file.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (writer != INVALID_HANDLE_VALUE) { CloseHandle(writer); throw std::runtime_error("expert file admitted concurrent mutation"); }
    require(GetLastError() == ERROR_SHARING_VIOLATION, "file write denial had an unrelated cause");
#else
    std::filesystem::resize_file(f.file, 1);
    std::array<uint8_t, 16> actual{};
    refused([&] { file.read_exact(0, actual.data(), actual.size()); });
    refused([&] { file.read_exact(0, actual.data(), 1); });
#endif
}

static void sealed_store() {
    fixture f;
    const expert_address a{0, 7, 64}, b{1, 91, 128}, absent{0, 999, 64};
    expert_lease retained;
    {
        memory_experts store(192, 2);
        refused([&] { store.acquire(a); });
        store.preload(a, f.data.data(), 64);
        refused([&] { store.preload({100, 0, 64}, f.data.data(), 64); });
        refused([&] { store.preload({0, UINT64_MAX, 64}, f.data.data(), 64); });
        refused([&] { store.preload({0, 0, 63}, f.data.data(), 64); });
        refused([&] { store.preload({0, 0, 64}, nullptr, 64); });
        refused([&] { store.preload(a, f.data.data(), 64); });
        refused([&] { store.seal({a, b}); });
        store.preload(b, f.data.data() + 64, 128);
        refused([&] { store.preload(absent, f.data.data(), 64); });
        refused([&] { store.seal({a, a}); });
        refused([&] { store.seal({a, absent}); });
        store.seal({b, a});
        require(store.payload_bytes() == 192, "memory-store payload accounting differs");
        refused([&] { store.preload(absent, f.data.data(), 64); });
        refused([&] { store.acquire(absent); });
        refused([&] { store.seal({a, b}); });
        std::vector<std::future<void>> readers;
        for (int i = 0; i < 8; ++i) {
            readers.push_back(std::async(std::launch::async, [&] {
                for (int j = 0; j < 100; ++j) {
                    const auto bytes = store.acquire(b);
                    require(bytes->size() == 128 && std::equal(bytes->begin(), bytes->end(), f.data.begin() + 64), "sealed expert lookup changed bytes");
                }
            }));
        }
        for (auto & reader : readers) { reader.get(); }
        retained = store.acquire(a);
    }
    require(retained->size() == 64 && std::equal(retained->begin(), retained->end(), f.data.begin()), "expert lease outlived its backing bytes");
    refused([] { memory_experts invalid(0, 1); });
    memory_experts bounded(16, 1);
    refused([&] { bounded.preload(a, f.data.data(), 64); });
}

int main() {
    try {
        file_reads(); concurrent_reads(); large_offsets(); invalid_reads(); changed_source(); sealed_store();
        std::puts("GLM storage: six suites passed (positional bytes, concurrency, 64-bit offsets, invalid spans, mutation, sealed leases)");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GLM storage test failed: %s\n", error.what());
        return 1;
    }
}
