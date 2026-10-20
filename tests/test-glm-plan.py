#!/usr/bin/env python3
"""Exercise the native planner with metadata-only GGUF files and malformed inputs."""

import argparse
import json
import pathlib
import struct
import subprocess
import tempfile
import unittest


def string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def scalar(key: str, value: int, kind: int = 4) -> bytes:
    formats = {2: "H", 4: "I", 5: "i"}
    return string(key) + struct.pack("<I" + formats[kind], kind, value)


def fixture(*, missing_projection: bool = False, wrong_experts: bool = False, kv_type: int = 5) -> bytes:
    values = {
        "block_count": 46, "nextn_predict_layers": 1, "embedding_length": 4096,
        "hyper_connection.count": 4, "attention.head_count": 64, "kda.head_dim": 128,
        "ssm.conv_kernel": 4, "attention.kv_lora_rank": 512, "attention.indexer.key_length": 128,
        "attention.indexer.kpool": 4, "expert_count": 288, "expert_used_count": 8,
        "context_length": 1048576,
    }
    fields = [string("general.architecture") + struct.pack("<I", 8) + string("glm5-next")]
    fields += [scalar("glm5-next." + key, value) for key, value in values.items()]
    heads = [int(i % 4 == 3 or i == 45) for i in range(46)]
    field = string("glm5-next.attention.head_count_kv") + struct.pack("<IIQ", 9, kv_type, 46)
    fields.append(field + struct.pack("<46" + ("i" if kv_type == 5 else "I"), *heads))
    tensors = [("token_embd.weight", (4096, 1)), ("output.weight", (4096, 1)), ("output_norm.weight", (4096,))]
    for layer in range(46):
        for projection in ("gate", "up", "down"):
            shape = (2048, 4096) if projection == "down" else (4096, 2048)
            if layer < 3:
                tensors.append((f"blk.{layer}.ffn_{projection}.weight", shape))
            else:
                if not (missing_projection and layer == 23 and projection == "up"):
                    count = 287 if wrong_experts and layer == 23 else 288
                    tensors.append((f"blk.{layer}.ffn_{projection}_exps.weight", shape + (count,)))
                tensors.append((f"blk.{layer}.ffn_{projection}_shexp.weight", shape))
    fields += [scalar("split.no", 0, 2), scalar("split.count", 1, 2), scalar("split.tensors.count", len(tensors), 5)]
    table = bytearray()
    offset = 0
    for name, shape in tensors:
        table += string(name) + struct.pack("<I", len(shape)) + struct.pack("<" + "q" * len(shape), *shape)
        table += struct.pack("<IQ", 0, offset)  # F32 descriptors, with no payload data.
        size = 4
        for dim in shape:
            size *= dim
        offset += (size + 31) // 32 * 32
    data = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(fields)) + b"".join(fields) + table
    return data + b"\0" * ((-len(data)) % 32)


class PlannerTests(unittest.TestCase):
    executable: pathlib.Path

    def run_planner(self, data: bytes, *args: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix="glm fixture ") as directory:
            path = pathlib.Path(directory) / "model header.gguf"
            path.write_bytes(data)
            return subprocess.run([str(self.executable), *args, str(path)], text=True, capture_output=True, timeout=30)

    def test_real_geometry_without_allocating_payload(self) -> None:
        result = self.run_planner(fixture())
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads(result.stdout)
        self.assertEqual(manifest["stages"][1]["dsa_layers"], 1)
        self.assertEqual(manifest["stages"][1]["mla_cache_bytes"], 1073741824)
        self.assertEqual(manifest["stages"][1]["index_cache_bytes"], 805306368)
        self.assertEqual(manifest["admission"], "NOT_VALIDATED")
        self.assertFalse(manifest["tensor_payloads_verified"])
        expert = next(t for t in manifest["tensors"] if t["name"] == "blk.23.ffn_up_exps.weight")
        self.assertEqual(expert["rank"], 1)
        self.assertEqual(expert["bytes_per_expert"], 33554432)

    def test_unsigned_head_schedule(self) -> None:
        result = self.run_planner(fixture(kv_type=4))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_middle_projection(self) -> None:
        result = self.run_planner(fixture(missing_projection=True))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing required", result.stderr)

    def test_wrong_expert_count(self) -> None:
        result = self.run_planner(fixture(wrong_experts=True))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid routed-expert", result.stderr)

    def test_truncated_header(self) -> None:
        result = self.run_planner(fixture()[:-40])
        self.assertNotEqual(result.returncode, 0)

    def test_malformed_split(self) -> None:
        for split in ("22,5,17", "22,0,23", "22,5,18,1", "-1,5,41"):
            with self.subTest(split=split):
                self.assertNotEqual(self.run_planner(fixture(), "--split", split).returncode, 0)

    def test_excessive_count_before_allocation(self) -> None:
        result = self.run_planner(b"GGUF" + struct.pack("<IQQ", 3, 2**63 - 1, 73))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("count exceeds planner limits", result.stderr)

    def test_context_refusal(self) -> None:
        self.assertNotEqual(self.run_planner(fixture(), "--context", "1048577").returncode, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=pathlib.Path, required=True)
    args, remaining = parser.parse_known_args()
    PlannerTests.executable = args.executable.resolve()
    unittest.main(argv=[__file__, *remaining])
