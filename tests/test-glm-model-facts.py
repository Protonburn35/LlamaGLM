#!/usr/bin/env python3
"""Audit exact real metadata headers; no tensor payload files are needed or read."""

import argparse
import copy
import hashlib
import json
import pathlib
import struct
import subprocess
import tempfile

PINS = [
    (9429984,9429956,"e47e6ba043a5d406cdf695219ff9728666dc3a0392eb7190171ca13fd671d63d"),
    (49261511584,38304,"ad6191c93269faeee16be8d2d1746df56f43e7f6515c26d646368a358f0ad7d4"),
    (49075832480,35232,"eed639822ed7b1e16efe6a4ab6d68728b5b3648c3270fe379218fb984097747e"),
    (22020797792,15328,"269b5ce802b9e424ad10d9781c709eace77e3310e5e1ba63939a3c074bfb20e3"),
]


def model_digest(report: dict, config: dict) -> str:
    data = bytearray(b"LlamaGLM execution identity v1\0"+struct.pack("<IHH",1,1,0))
    def text(value: str) -> None:
        raw = value.encode("ascii")
        data.extend(struct.pack("<I",len(raw))+raw)
    for value in (report["source_revision"],"glm5-next",config["model"]["repository"],config["model"]["revision"],config["model"]["quant"]):
        text(value)
    data.extend(struct.pack("<IIIQ",1,1,1412,120358051192))
    data.extend(struct.pack("<IIIIIIII",45,1,288,8,1,*report["layer_ends"]))
    data.extend(struct.pack("<IIIIIIQ",4096,4,report["row_limit"],report["context_tokens"],report["frame_bytes"],16<<10,128<<20))
    for value in ("f32le:hidden,stream,row","f16:mla,index","f32:kda","causal:seq0:append-only"):
        text(value)
    data.extend(struct.pack("<II",0,4))
    for i,(size,header,fingerprint) in enumerate(PINS):
        data.extend(struct.pack("<I",i)); text(f"GLM-5.3-Flash-UD-IQ3_XXS-{i+1:05d}-of-00004.gguf")
        data.extend(struct.pack("<QQ",size,header)+bytes.fromhex(fingerprint))
    return hashlib.sha256(data).hexdigest()


def run(executable: pathlib.Path, header_directory: pathlib.Path) -> None:
    headers = [header_directory / f"GLM-5.3-Flash-UD-IQ3_XXS-{i+1:05d}-of-00004.gguf.header" for i in range(4)]
    for path,(_,size,fingerprint) in zip(headers,PINS):
        if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != fingerprint:
            raise RuntimeError("reference headers do not match the audited metadata-only pins")
    example = pathlib.Path(__file__).resolve().parents[1] / "docs/config/deployment.example.json"
    initial = json.loads(example.read_text())
    with tempfile.TemporaryDirectory(prefix="glm metadata facts \u03bb ") as directory:
        root = pathlib.Path(directory)
        config_file = root / "NEVER-ECHO-PRIVATE-CONFIG.json"
        def invoke(config: dict, paths: list[pathlib.Path], success: bool) -> dict | None:
            config_file.write_text(json.dumps(config))
            result = subprocess.run([str(executable),"--audit-headers",str(config_file),*map(str,paths)],
                                    capture_output=True,text=True,timeout=30)
            if (result.returncode == 0) != success or "NEVER-ECHO" in result.stdout+result.stderr:
                raise RuntimeError("metadata inspection failed or exposed a local path")
            if not success:
                if result.stdout:
                    raise RuntimeError("failed metadata inspection emitted partial validated facts")
                return None
            report = json.loads(result.stdout)
            if not (report["configuration"] == "VALID" and report["admission"] == "NOT_VALIDATED" and
                    report["tensor_schema"] == "VALIDATED" and report["schema_checked_tensors"] == 1412 and
                    report["schema_weight_allocation_bytes"] == report["model_payload_bytes_read"] == report["network_operations"] == 0 and
                    report["metadata_bytes_read"] == 9518820 and report["full_file_sizes_checked"] is False and
                    report["reference_headers_only"] is True and report["tensor_payloads_verified"] is False and
                    report["credentials_read"] is False and report["model_identity_sha256"] == model_digest(report,config)):
                raise RuntimeError("metadata facts violated native schema, independent identity, bounded reads or reference-only status")
            for i,(shard,(size,header,fingerprint)) in enumerate(zip(report["shards"],PINS)):
                if (shard["number"],shard["file_bytes"],shard["header_bytes"],shard["header_sha256"]) != (i,size,header,fingerprint):
                    raise RuntimeError("metadata facts differ from exact pinned shard evidence")
            return report
        first = invoke(initial,headers,True)
        reordered = copy.deepcopy(initial); reordered["rank"] = 2; reordered["nodes"].reverse()
        second = invoke(reordered,headers,True)
        if (first["intent_sha256"],first["model_identity_sha256"]) != (second["intent_sha256"],second["model_identity_sha256"]):
            raise RuntimeError("local rank/node input order changed common checked facts")
        explicit = copy.deepcopy(initial)
        explicit.update(context=4096,row_limit=64,frame_bytes=65536,layer_ends=[20,26,45])
        changed = invoke(explicit,headers,True)
        if changed["model_identity_sha256"] == first["model_identity_sha256"] or changed["intent_sha256"] == first["intent_sha256"]:
            raise RuntimeError("capacity/split change was not bound into both checked identities")
        swapped = list(headers); swapped[1],swapped[2] = swapped[2],swapped[1]
        invoke(initial,swapped,False)
        altered = bytearray(headers[1].read_bytes()); altered[64] ^= 1
        private = root / "NEVER-ECHO-PRIVATE-HEADER.header"; private.write_bytes(altered)
        replacements = list(headers); replacements[1] = private
        invoke(initial,replacements,False)
        private.write_bytes(altered[:-1]); invoke(initial,replacements,False)
        replacements[1] = root / "NEVER-ECHO-MISSING.header"; invoke(initial,replacements,False)
    print("PINNED_MODEL_HEADER_FACTS_PASS: three independent identity/schema cases, four malformed/source refusals; 9518820 metadata bytes; zero tensor/credential/network access; full-file/payload/residency NOT_VALIDATED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable",required=True,type=pathlib.Path)
    parser.add_argument("--headers",required=True,type=pathlib.Path)
    args = parser.parse_args()
    run(args.executable.resolve(strict=True),args.headers.resolve(strict=True))
