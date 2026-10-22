// Small pre-flight check for RPC workers: for every device of every endpoint, upload quantized weights,
// run MUL_MAT and MUL_MAT_ID on the remote device and compare the result with the local CPU backend.
// It allocates a few MiB per device, so it is safe to run before loading a large model.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-rpc.h"

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

typedef ggml_backend_reg_t (*rpc_add_server_t)(const char * endpoint);
typedef void (*rpc_get_telemetry_t)(uint64_t * weight_bytes, uint32_t * weight_tensors, uint64_t * act_sent, uint64_t * act_recv, uint64_t * graphs_computed);

struct expect_range {
    double min_gib;
    double max_gib;
};

static void print_usage(const char * prog) {
    printf("usage: %s [options] endpoint [endpoint ...]\n\n", prog);
    printf("options:\n");
    printf("  --devices N                      every endpoint must report exactly N devices\n");
    printf("  --expect ENDPOINT/DEV=MIN:MAX    total memory of device DEV of ENDPOINT must be in [MIN, MAX] GiB\n");
    printf("  --types LIST                     weight types to test (default: iq2_s,iq3_s,iq4_xs,q6_K,q8_0,f16)\n");
    printf("\nexample:\n");
    printf("  %s --devices 2 --expect 192.168.50.129:50052/0=6:9 --expect 192.168.50.129:50052/1=24:40 192.168.50.129:50052\n", prog);
}

static std::vector<float> rand_vec(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto & x : v) {
        x = dist(rng);
    }
    return v;
}

// normalized mean squared error of a vs. reference b
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double err = 0.0;
    double ref = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i])) {
            return INFINITY;
        }
        err += (double) (a[i] - b[i]) * (a[i] - b[i]);
        ref += (double) b[i] * b[i];
    }
    return ref > 0.0 ? err / ref : err;
}

// one MUL_MAT (n_as == 0) or MUL_MAT_ID (n_as > 0) case, computed on the given backend
static bool run_case(ggml_backend_t backend, ggml_type type, int64_t k, int64_t m, int64_t n, int64_t n_as, int64_t n_used,
                     const std::vector<uint8_t> & wq, const std::vector<float> & x, const std::vector<int32_t> & ids,
                     std::vector<float> & out) {
    ggml_init_params params = {
        /*.mem_size   =*/ 8*ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(params) };

    ggml_tensor * w = ggml_new_tensor_3d(ctx.get(), type, k, m, n_as > 0 ? n_as : 1);
    ggml_tensor * b = nullptr;
    ggml_tensor * t_ids = nullptr;
    ggml_tensor * y = nullptr;
    if (n_as > 0) {
        b     = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, n_used, n);
        t_ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_used, n);
        y     = ggml_mul_mat_id(ctx.get(), w, b, t_ids);
    } else {
        b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, n);
        y = ggml_mul_mat(ctx.get(), w, b);
    }

    ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors(ctx.get(), backend) };
    if (!buf) {
        printf("      allocation failed on %s\n", ggml_backend_name(backend));
        return false;
    }
    ggml_backend_tensor_set(w, wq.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(b, x.data(), 0, ggml_nbytes(b));
    if (t_ids) {
        ggml_backend_tensor_set(t_ids, ids.data(), 0, ggml_nbytes(t_ids));
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, y);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("      graph compute failed on %s\n", ggml_backend_name(backend));
        return false;
    }
    out.resize(ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    return true;
}

static bool test_device(ggml_backend_dev_t dev, ggml_backend_t cpu, const std::vector<ggml_type> & types) {
    ggml_backend_ptr backend { ggml_backend_dev_init(dev, nullptr) };
    if (!backend) {
        printf("    [FAIL] could not create a backend for %s\n", ggml_backend_dev_name(dev));
        return false;
    }

    const int64_t k = 2048;
    const int64_t m = 256;
    const int64_t n_as = 8;
    const int64_t n_used = 2;

    bool ok = true;
    for (ggml_type type : types) {
        for (int64_t n_expert : {int64_t(0), n_as}) {
            for (int64_t n : {int64_t(1), int64_t(8)}) {
                const int64_t n_mat = n_expert > 0 ? n_expert : 1;
                const std::vector<float> wf = rand_vec(k*m*n_mat, 1234 + (uint32_t) type);
                std::vector<uint8_t> wq(ggml_row_size(type, k)*m*n_mat);
                ggml_quantize_chunk(type, wf.data(), wq.data(), 0, m*n_mat, k, nullptr);

                const std::vector<float> x = rand_vec(k*n*(n_expert > 0 ? n_used : 1), 42);
                std::vector<int32_t> ids(n_used*n);
                for (int64_t i = 0; i < n; i++) {
                    for (int64_t j = 0; j < n_used; j++) {
                        ids[i*n_used + j] = (int32_t) ((i*3 + j*5) % n_as);
                    }
                }

                std::vector<float> y_dev;
                std::vector<float> y_ref;
                const auto t0 = std::chrono::steady_clock::now();
                const bool ran = run_case(backend.get(), type, k, m, n, n_expert, n_used, wq, x, ids, y_dev);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (!ran || !run_case(cpu, type, k, m, n, n_expert, n_used, wq, x, ids, y_ref)) {
                    ok = false;
                    continue;
                }
                const double err = nmse(y_dev, y_ref);
                const bool pass = err < 5e-4;
                ok = ok && pass;
                printf("    %-11s %-6s n=%" PRId64 ": nmse %.2e, %.1f ms -> %s\n", n_expert > 0 ? "MUL_MAT_ID" : "MUL_MAT",
                       ggml_type_name(type), n, err, ms, pass ? "ok" : "FAIL");
            }
        }
    }
    return ok;
}

int main(int argc, char ** argv) {
    std::vector<std::string> endpoints;
    std::map<std::string, expect_range> expect;
    int n_devices_expected = -1;
    std::vector<ggml_type> types = { GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_F16 };

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--devices" && i + 1 < argc) {
            n_devices_expected = std::atoi(argv[++i]);
        } else if (arg == "--expect" && i + 1 < argc) {
            const std::string v = argv[++i];
            const size_t eq = v.find('=');
            const size_t colon = v.find(':', eq);
            if (eq == std::string::npos || colon == std::string::npos) {
                fprintf(stderr, "invalid --expect value: %s\n", v.c_str());
                return 1;
            }
            expect[v.substr(0, eq)] = { std::atof(v.substr(eq + 1, colon - eq - 1).c_str()), std::atof(v.substr(colon + 1).c_str()) };
        } else if (arg == "--types" && i + 1 < argc) {
            types.clear();
            std::string list = argv[++i];
            size_t pos = 0;
            while (pos <= list.size()) {
                const size_t end = std::min(list.find(',', pos), list.size());
                const std::string name = list.substr(pos, end - pos);
                bool found = false;
                for (int t = 0; t < GGML_TYPE_COUNT; t++) {
                    const char * tn = ggml_type_name((ggml_type) t);
                    if (tn && name == tn) {
                        types.push_back((ggml_type) t);
                        found = true;
                    }
                }
                if (!found) {
                    fprintf(stderr, "unknown type: %s\n", name.c_str());
                    return 1;
                }
                pos = end + 1;
            }
        } else if (arg.rfind("-", 0) != 0) {
            endpoints.push_back(arg);
        } else {
            fprintf(stderr, "unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    if (endpoints.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        fprintf(stderr, "RPC backend not found, ggml-rpc must be next to this program\n");
        return 1;
    }
    auto add_server = (rpc_add_server_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    auto get_telemetry = (rpc_get_telemetry_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_get_telemetry");
    if (!add_server) {
        fprintf(stderr, "ggml_backend_rpc_add_server not found in the RPC backend\n");
        return 1;
    }
    ggml_backend_ptr cpu { ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr) };
    if (!cpu) {
        fprintf(stderr, "CPU backend not found\n");
        return 1;
    }

    bool all_ok = true;
    for (const std::string & ep : endpoints) {
        printf("%s\n", ep.c_str());
        ggml_backend_reg_t reg = add_server(ep.c_str());
        const size_t n_dev = reg ? ggml_backend_reg_dev_count(reg) : 0;
        if (n_dev == 0) {
            printf("  [FAIL] no devices: is ggml-rpc-server running there and is the port open in the firewall?\n");
            printf("GATE1 %s devices=0 result=FAIL\n", ep.c_str());
            all_ok = false;
            continue;
        }
        bool ep_ok = true;
        if (n_devices_expected >= 0 && (int) n_dev != n_devices_expected) {
            printf("  [FAIL] %zu devices reported, %d expected (did the worker fall back to CPU or miss -d?)\n", n_dev, n_devices_expected);
            ep_ok = false;
        }
        for (size_t i = 0; i < n_dev; i++) {
            ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
            size_t free = 0;
            size_t total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            const double total_gib = total/1073741824.0;
            printf("  device %zu (%s): total %.2f GiB, free %.2f GiB\n", i, ggml_backend_dev_name(dev), total_gib, free/1073741824.0);

            bool dev_ok = true;
            const auto it = expect.find(ep + "/" + std::to_string(i));
            if (it != expect.end() && (total_gib < it->second.min_gib || total_gib > it->second.max_gib)) {
                printf("    [FAIL] total memory %.2f GiB is outside the expected %.1f-%.1f GiB (wrong device type?)\n",
                       total_gib, it->second.min_gib, it->second.max_gib);
                dev_ok = false;
            }
            dev_ok = test_device(dev, cpu.get(), types) && dev_ok;
            printf("GATE1 %s dev=%zu total_mib=%zu free_mib=%zu result=%s\n", ep.c_str(), i, total/1048576, free/1048576, dev_ok ? "PASS" : "FAIL");
            ep_ok = ep_ok && dev_ok;
        }
        all_ok = all_ok && ep_ok;
    }

    if (get_telemetry) {
        uint64_t w = 0, sent = 0, recv = 0, graphs = 0;
        uint32_t wt = 0;
        get_telemetry(&w, &wt, &sent, &recv, &graphs);
        printf("client totals: sent %.2f MiB, received %.2f MiB, remote graphs %" PRIu64 "\n", sent/1048576.0, recv/1048576.0, graphs);
    }
    printf("GATE1 %s\n", all_ok ? "PASSED" : "FAILED");
    return all_ok ? 0 : 1;
}
