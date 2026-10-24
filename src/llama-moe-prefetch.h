#pragma once

#include "ggml.h"

// optional OS read-ahead of the selected experts of MoE layers whose experts are in host memory (usually mmap)
// enabled with the environment variable LLAMA_MOE_PREFETCH=1, results are not changed

bool llama_moe_prefetch_enabled();

// CPU op that asks the OS to read the experts selected by ids from the expert tensors exps
// returns nullptr if prefetch is disabled or an expert tensor is not a host weight
// add it to the graph before the matmuls of these experts
ggml_tensor * llama_moe_prefetch(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * const * exps, int n_exps);
