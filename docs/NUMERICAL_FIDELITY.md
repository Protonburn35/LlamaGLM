# Numerical contract and evidence

The target is the exact published GLM-5.3-Flash UD-IQ3_XXS tensor inventory. The planner preserves actual tensor types, assigns all trunk layers once and rejects expert slabs whose third dimension differs from 288. Geometry explicitly requires eight selected experts, four mHC streams, the audited 34 KDA/11 DSA schedule and one distinct NextN block. It never chooses experts or computes model outputs.

The existing CPU architecture fixture uses synthetic small tensors and verifies CPU logits, model-save roundtrip and mixed/chunked batch behavior. This is a baseline for the inherited evaluator, not a real-quant or distributed numerical oracle. The new ownership/state test independently checks byte arithmetic through 128K, 256K, 512K and 1M counters. It does not execute 1M inference.

Required next evidence: monolithic versus owned-layer intermediate mHC tensors and logits, independent KDA recurrence/convolution and DSA tail-pool fixtures, every actual quant codec on CPU/CUDA, all eight routed experts and the shared expert, cache transition equivalence, sampled target distributions, and coherent speculative rollback. CUDA fast-math and CPU activation quantization inherited from upstream need explicit qualification; no new precision shortcut is authorized by passing planner tests.

Current modes: FP32 boundary planning; FP16 sparse/index planning; source-faithful FP32 KDA state arithmetic. Optional BF16 transport, quantized caches, sampled MTP and vision execution are not implemented or validated by this project checkpoint.
