#include "stage-boundary.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace glm_rpc {
namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "GLM boundary requires IEEE binary32");

bool little_endian() {
    const uint32_t value = 1;
    return *reinterpret_cast<const uint8_t *>(&value) == 1;
}

void check_deadline(deadline until) {
    if (std::chrono::steady_clock::now() >= until) { throw io_error(io_failure::timeout, 0, "GLM boundary deadline expired"); }
}

header prepare(header value, const limits & bounds) {
    bounds.validate();
    if (value.type != kind::boundary || value.sender > 1 || value.receiver != value.sender + 1 ||
            value.hidden != bounds.hidden || value.streams != bounds.streams || value.element != dtype::f32 ||
            value.rows == 0 || value.rows > bounds.rows || value.position >= bounds.context ||
            value.rows > bounds.context - value.position || value.epoch == 0 || value.transaction == 0 || value.generation == 0 ||
            std::all_of(value.session.begin(), value.session.end(), [](uint8_t byte) { return byte == 0; })) {
        throw std::invalid_argument("invalid GLM boundary execution coordinates");
    }
    value.total_bytes = uint64_t(value.rows) * bounds.hidden * bounds.streams * sizeof(float);
    if (value.total_bytes > max_chunk_bytes) { throw std::invalid_argument("GLM boundary exceeds logical limit"); }
    value.fragments = uint32_t((value.total_bytes + bounds.frame_bytes - 1) / bounds.frame_bytes);
    value.fragment = value.payload_bytes = 0;
    return value;
}

bool same_chunk(const header & a, const header & b) {
    return a.type == b.type && a.sender == b.sender && a.receiver == b.receiver && a.session == b.session &&
        a.epoch == b.epoch && a.transaction == b.transaction && a.chunk == b.chunk && a.position == b.position &&
        a.generation == b.generation && a.rows == b.rows && a.hidden == b.hidden && a.streams == b.streams &&
        a.element == b.element && a.fragments == b.fragments && a.total_bytes == b.total_bytes;
}

void unpack(float * output, bytes_view input) {
    if (little_endian()) { std::memcpy(output, input.data, input.size); return; }
    for (size_t i = 0; i < input.size / sizeof(float); ++i) {
        const auto * p = input.data + i * sizeof(float);
        const uint32_t bits = uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
        std::memcpy(output + i, &bits, sizeof(bits));
    }
}

} // namespace

void send_boundary(tcp_socket & socket, sender & codec, header coordinates, const limits & bounds,
                   const float * values, size_t count, deadline until) {
    try {
        check_deadline(until);
        auto metadata = prepare(coordinates, bounds);
        if (!values || count != metadata.total_bytes / sizeof(float)) {
            throw std::invalid_argument("GLM boundary float buffer size differs from execution plan");
        }
        std::vector<uint8_t> encoded;
        if (!little_endian()) { encoded.resize(bounds.frame_bytes); }
        for (uint32_t i = 0; i < metadata.fragments; ++i) {
            const size_t offset = size_t(i) * bounds.frame_bytes;
            const size_t size = size_t(std::min<uint64_t>(bounds.frame_bytes, metadata.total_bytes - offset));
            const auto * data = reinterpret_cast<const uint8_t *>(values) + offset;
            if (!little_endian()) {
                for (size_t j = 0; j < size / sizeof(float); ++j) {
                    uint32_t bits;
                    std::memcpy(&bits, values + offset / sizeof(float) + j, sizeof(bits));
                    for (size_t byte = 0; byte < sizeof(bits); ++byte) { encoded[j * sizeof(bits) + byte] = uint8_t(bits >> (8 * byte)); }
                }
                data = encoded.data();
            }
            metadata.fragment = i;
            write_frame(socket, codec, metadata, {data, size}, until);
            check_deadline(until);
        }
    } catch (...) { socket.interrupt(); throw; }
}

void send_boundary_bytes(tcp_socket & socket, sender & codec, header coordinates, const limits & bounds,
                         bytes_view payload, deadline until) {
    try {
        check_deadline(until);
        auto metadata = prepare(coordinates, bounds);
        if (!payload.data || payload.size != metadata.total_bytes) {
            throw std::invalid_argument("canonical boundary bytes differ from the execution plan");
        }
        for (uint32_t i = 0; i < metadata.fragments; ++i) {
            const size_t offset = size_t(i) * bounds.frame_bytes;
            const size_t size = size_t(std::min<uint64_t>(bounds.frame_bytes, metadata.total_bytes - offset));
            metadata.fragment = i;
            write_frame(socket, codec, metadata, {payload.data + offset,size}, until);
            check_deadline(until);
        }
    } catch (...) { socket.interrupt(); throw; }
}

std::vector<float> receive_boundary(tcp_socket & socket, receiver & codec, header expected, const limits & bounds,
                                    uint64_t input_budget, deadline until) {
    try {
        check_deadline(until);
        expected = prepare(expected, bounds);
        if (input_budget == 0 || input_budget > max_chunk_bytes || expected.total_bytes > input_budget) {
            throw std::invalid_argument("GLM boundary input exceeds the local reserved budget");
        }
        std::vector<float> input;
        for (uint32_t i = 0; i < expected.fragments; ++i) {
            const auto message = read_frame(socket, codec, until);
            const size_t offset = size_t(i) * bounds.frame_bytes;
            const uint64_t size = std::min<uint64_t>(bounds.frame_bytes, expected.total_bytes - offset);
            if (!same_chunk(message.metadata, expected) || message.metadata.fragment != i || message.payload.size() != size) {
                throw std::invalid_argument("GLM boundary fragments differ from the active execution plan");
            }
            if (i == 0) { input.resize(size_t(expected.total_bytes / sizeof(float))); }
            unpack(input.data() + offset / sizeof(float), {message.payload.data(), message.payload.size()});
            check_deadline(until);
        }
        return input;
    } catch (...) { socket.interrupt(); throw; }
}

} // namespace glm_rpc
