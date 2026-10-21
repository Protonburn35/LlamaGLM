#!/usr/bin/env python3
"""Check local deployment syntax/privacy without model, credential or network access."""

import argparse
import copy
import hashlib
import json
import pathlib
import subprocess
import struct
import tempfile


def run(executable: pathlib.Path) -> None:
    example = pathlib.Path(__file__).resolve().parents[1] / "docs/config/deployment.example.json"
    initial = json.loads(example.read_text())
    private = "NEVER-ECHO-THIS-PRIVATE-CONFIG-VALUE"
    initial["secret_file"] = f"credentials/{private}.bin"
    with tempfile.TemporaryDirectory(prefix="glm local config \u03bb ") as directory:
        path = pathlib.Path(directory) / "deployment.json"

        def invoke(value: dict | str, success: bool) -> dict | None:
            text = value if isinstance(value, str) else json.dumps(value)
            path.write_text(text)
            result = subprocess.run([str(executable), "--check", str(path)], capture_output=True, text=True, timeout=5)
            if (result.returncode == 0) != success:
                raise RuntimeError(f"configuration exit {result.returncode}: {result.stderr}")
            if private in result.stdout + result.stderr or str(path) in result.stdout + result.stderr:
                raise RuntimeError("configuration output exposed local paths or private values")
            return json.loads(result.stdout) if success else None

        for rank in range(3):
            value = copy.deepcopy(initial)
            value["rank"] = rank
            value["nodes"].reverse()
            report = invoke(value, True)
            encoded = bytearray(b"LlamaGLM deployment intent v1\0" + struct.pack("<I",1))
            for text in (value["model"][key] for key in ("repository","revision","quant")):
                data = text.encode("ascii")
                encoded += struct.pack("<I",len(data)) + data
            encoded += struct.pack("<IIIIII",1048576,2048,4<<20,22,27,45)
            for node in sorted(value["nodes"],key=lambda item:item["rank"]):
                data = node["address"].encode("ascii")
                encoded += struct.pack("<II",node["rank"],len(data))+data
                encoded += struct.pack("<III",node["data_port"],node["control_port"],node["api_port"])
            if not (report["configuration"] == "VALID" and report["admission"] == "NOT_VALIDATED" and
                    report["rank"] == rank and report["context_tokens"] == 1048576 and
                    report["layer_ends"] == [22, 27, 45] and report["local_model_files"] == 4 and
                    report["model_payload_bytes_read"] == report["network_operations"] == 0 and
                    report["credentials_read"] is False and report["tensor_schema"] == "NOT_RUN" and
                    [node["rank"] for node in report["nodes"]] == [0, 1, 2] and len(report["source_revision"]) == 40 and
                    report["intent_sha256"] == hashlib.sha256(encoded).hexdigest()):
                raise RuntimeError("local deployment validation claimed admission or changed target defaults")
        value = copy.deepcopy(initial)
        value.update({"context": 4096, "row_limit": 64, "frame_bytes": 65536, "layer_ends": [20, 26, 45]})
        report = invoke(value, True)
        if (report["context_tokens"], report["row_limit"], report["frame_bytes"], report["layer_ends"]) != (4096, 64, 65536, [20, 26, 45]):
            raise RuntimeError("explicit local capacity/split values changed")
        baseline = invoke(initial,True)["intent_sha256"]
        relocated = copy.deepcopy(initial)
        relocated["model"]["files"] = [str(pathlib.Path(directory) / "different local model root" / pathlib.Path(name).name)
                                        for name in initial["model"]["files"]]
        relocated["secret_file"] = f"credentials/\u03bb/{private}.bin"
        if invoke(relocated,True)["intent_sha256"] != baseline:
            raise RuntimeError("local paths changed the common three-owner deployment intent")
        for change in (lambda v: v.update(context=4096), lambda v: v.update(row_limit=64),
                       lambda v: v.update(frame_bytes=65536), lambda v: v.update(layer_ends=[20,26,45]),
                       lambda v: v["nodes"][1].update(address="192.168.50.21"),
                       lambda v: v["nodes"][1].update(data_port=50053)):
            value = copy.deepcopy(initial); change(value)
            if invoke(value,True)["intent_sha256"] == baseline:
                raise RuntimeError("shared deployment intent omitted a topology/capacity/split field")

        mutations = [
            lambda v: v.update(version=2), lambda v: v.update(version=True), lambda v: v.update(rank=3),
            lambda v: v.update(rank=-1), lambda v: v.update(context=0), lambda v: v.update(context=1048577),
            lambda v: v.update(row_limit=2049), lambda v: v.update(frame_bytes=65537),
            lambda v: v.update(layer_ends=[22, 22, 45]), lambda v: v.update(layer_ends=[22, 27, 44]),
            lambda v: v.update(nodes=v["nodes"][:2]), lambda v: v["nodes"][1].update(rank=0),
            lambda v: v["nodes"][1].update(address=v["nodes"][0]["address"]),
            lambda v: v["nodes"][1].update(address="127.0.0.1"),
            lambda v: v["nodes"][1].update(address="0.0.0.0"),
            lambda v: v["nodes"][1].update(address="224.0.0.1"),
            lambda v: v["nodes"][1].update(address="192.168.050.11"),
            lambda v: v["nodes"][1].update(address="localhost"),
            lambda v: v["nodes"][1].update(data_port=0),
            lambda v: v["nodes"][1].update(control_port=50051),
            lambda v: v["nodes"][2].update(api_port=8000),
            lambda v: v["nodes"][0].update(control_port=50052),
            lambda v: v["nodes"][0].update(api_port=65536),
            lambda v: v["model"].update(quant="Q3_K"),
            lambda v: v["model"].update(revision="0" * 40),
            lambda v: v["model"].update(files=v["model"]["files"][:3]),
            lambda v: v["model"]["files"].reverse(),
            lambda v: v["model"].update(extra=True), lambda v: v.update(secret=private),
            lambda v: v.update(secret_file="https://invalid.example/" + private),
            lambda v: v.update(secret_file="\\\\server\\" + private),
            lambda v: v.update(secret_file="."), lambda v: v.update(secret_file="bad\u0000" + private),
            lambda v: v.pop("secret_file"), lambda v: v.update(secret_file=23),
        ]
        for change in mutations:
            value = copy.deepcopy(initial)
            change(value)
            invoke(value, False)
        invoke('{"rank":0,"rank":1,"private":"' + private + '"}', False)
        invoke('{"secret_file":"' + private, False)
        invoke('[' * 20 + '0' + ']' * 20, False)
        invoke(json.dumps(initial) + " " * 65536, False)
        version = subprocess.run([str(executable), "--version"], capture_output=True, text=True, timeout=5)
        if version.returncode or "deployment schema 1" not in version.stdout:
            raise RuntimeError("deployment executable did not report its schema/source")
        print("GLM_DEPLOYMENT_CONFIG_PASS: three owners/defaults, explicit capacities, 39 malformed/privacy refusals; no model/credential files or network operations; admission NOT_VALIDATED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=pathlib.Path, required=True)
    args = parser.parse_args()
    run(args.executable.resolve(strict=True))
