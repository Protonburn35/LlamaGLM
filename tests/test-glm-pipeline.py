#!/usr/bin/env python3
"""Qualify three actual owned-layer CPU processes against a separate monolithic oracle."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import queue
import secrets
import shutil
import struct
import subprocess
import tempfile
import threading


class Worker:
    def __init__(self, executable: pathlib.Path, data: pathlib.Path, key_file: pathlib.Path, output: pathlib.Path,
                 rank: int, next_port: int = 0, *, streamed: bool = False):
        self.rank = rank
        self.output = output
        self.log_path = output.with_suffix(".log")
        self.log = self.log_path.open("wb")
        command = [str(executable), "--rank", str(rank), "--model", str(data / "model.gguf"),
                   "--requests", str(data / "requests.json"), "--key-file", str(key_file), "--output", str(output),
                   "--storage", "streamed" if streamed else "buffered"]
        if next_port:
            command += ["--next-port", str(next_port)]
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        except Exception:
            self.log.close()
            raise
        self.lines: list[str] = []
        self.events: queue.Queue[str | None] = queue.Queue(maxsize=16)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.append(line.strip())
            self.events.put(line.strip())
        self.events.put(None)

    def listen_port(self) -> int:
        while True:
            line = self.events.get(timeout=45)
            if line is None:
                raise RuntimeError(f"rank {self.rank} exited before listening: {self.log_path.read_text(errors='replace')[-4000:]}")
            fields = line.split()
            if fields[:1] == ["GLM_PIPELINE_LISTEN"]:
                if len(fields) != 3 or int(fields[1]) != self.rank or not 0 < int(fields[2]) < 65536:
                    raise RuntimeError("invalid fixture listener announcement")
                return int(fields[2])

    def finish(self, success: bool) -> None:
        code = self.process.wait(timeout=45)
        self.reader.join(timeout=2)
        if self.reader.is_alive():
            raise RuntimeError("worker output reader did not finish")
        if (code == 0) != success:
            raise RuntimeError(f"rank {self.rank} exit {code}: {self.log_path.read_text(errors='replace')[-4000:]}")

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.reader.join(timeout=2)
        if self.process.stdout:
            self.process.stdout.close()
        self.log.close()


def compare(expected: pathlib.Path, actual: pathlib.Path, width: int, exact: bool) -> float:
    reference = expected.read_bytes()
    computed = actual.read_bytes()
    if len(reference) != 131 * width * 4 or len(computed) != len(reference):
        raise RuntimeError("missing or incorrectly sized pipeline outputs")
    if exact and reference != computed:
        raise RuntimeError("four-stream pipeline boundary differs from monolithic oracle bytes")
    maximum = 0.0
    for (a,), (b,) in zip(struct.iter_unpack("<f", reference), struct.iter_unpack("<f", computed)):
        error = abs(a - b)
        if not math.isfinite(a) or not math.isfinite(b) or error > 2e-5 * (1 + abs(a)):
            raise RuntimeError("pipeline logits differ from monolithic oracle")
        maximum = max(maximum, error)
    return maximum


def check_storage(worker: Worker, streamed: bool, logical_bytes: int) -> None:
    records = [json.loads(line.removeprefix("GLM_PIPELINE_STORAGE ")) for line in worker.lines
               if line.startswith("GLM_PIPELINE_STORAGE ")]
    if len(records) != 2 or [record["phase"] for record in records] != ["startup", "complete"]:
        raise RuntimeError("worker storage evidence is missing or out of order")
    desktop = streamed and worker.rank != 1
    for record in records:
        if (record["version"], record["rank"], record["microbatch"], record["mode"]) != (
                1, worker.rank, 1 if streamed else 32, "desktop-streamed" if desktop else "buffered"):
            raise RuntimeError("worker executed the wrong storage/microbatch profile")
        if desktop:
            if not (0 < record["routed_logical_bytes"] < logical_bytes and
                    0 < record["model_buffer_bytes"] < logical_bytes and
                    record["streamed_tensors"] == (3 if worker.rank == 0 else 6) and
                    0 <= record["host_reserved_bytes"] <= 16 << 20):
                raise RuntimeError("streamed worker did not defer experts or exceeded its host payload cap")
            if record["phase"] == "startup":
                if any(record[key] != 0 for key in ("native_read_requests", "native_read_bytes", "host_reserved_bytes")):
                    raise RuntimeError("streamed worker read expert payloads during startup")
            elif record["native_read_requests"] <= 0 or record["native_read_bytes"] <= 0:
                raise RuntimeError("streamed model worker performed no source reads")
        elif any(key.startswith("native_read") for key in record):
            raise RuntimeError("buffered worker unexpectedly exposed a desktop disk source")
    if desktop and any(records[0][key] != records[1][key] for key in
                       ("routed_logical_bytes", "model_buffer_bytes", "streamed_tensors")):
        raise RuntimeError("streamed worker changed weight ownership during inference")
    if desktop:
        final = records[1]
        print(f"STREAMED_SOURCE_RANK {worker.rank}: routed logical {final['routed_logical_bytes']}; "
              f"ordinary buffers {final['model_buffer_bytes']}; host reserved {final['host_reserved_bytes']}; "
              f"native reads {final['native_read_requests']}; native requested bytes {final['native_read_bytes']}")


def run(oracle: pathlib.Path, executable: pathlib.Path, *, streamed: bool = False) -> None:
    with tempfile.TemporaryDirectory(prefix="glm three process fixture ") as directory:
        root = pathlib.Path(directory)
        data = root / "oracle and model"
        command = [str(oracle), "--glm-pipeline-fixture", str(data), "--seed", "53"]
        if streamed:
            command += ["--glm-pipeline-streamed"]
        generation = subprocess.run(command,
                                    capture_output=True, text=True, timeout=60)
        if generation.returncode != 0 or "GLM pipeline oracle:" not in generation.stdout:
            raise RuntimeError(f"oracle generation failed: {generation.stdout[-2000:]} {generation.stderr[-4000:]}")
        requests = json.loads((data / "requests.json").read_text())
        if requests["microbatch"] != (1 if streamed else 32):
            raise RuntimeError("independent monolithic oracle used the wrong microbatch")
        key_file = root / "ephemeral-fixture-key.bin"
        key_file.touch(mode=0o600, exist_ok=False)
        key_file.write_bytes(secrets.token_bytes(32))
        workers: list[Worker] = []
        try:
            c = Worker(executable, data, key_file, root / "stage-C.f32le", 2, streamed=streamed)
            workers.append(c)
            b = Worker(executable, data, key_file, root / "stage-B.f32le", 1, c.listen_port(), streamed=streamed)
            workers.append(b)
            a = Worker(executable, data, key_file, root / "stage-A.f32le", 0, b.listen_port(), streamed=streamed)
            workers.append(a)
            if len({w.process.pid for w in workers}) != 3:
                raise RuntimeError("pipeline did not create three distinct worker processes")
            for worker in workers:
                worker.finish(True)
            payload_bytes = 0
            maximum = 0.0
            for worker in workers:
                owned = [line.split() for line in worker.lines if line.startswith("GLM_PIPELINE_OWNED ")]
                done = [line.split() for line in worker.lines if line.startswith("GLM_PIPELINE_DONE ")]
                if len(owned) != 1 or len(owned[0]) != 5 or list(map(int, owned[0][1:4])) != [worker.rank, 2*worker.rank, 2*worker.rank+2]:
                    raise RuntimeError("worker did not report its exact owned layer range")
                if len(done) != 1 or list(map(int, done[0][1:])) != [worker.rank, 131]:
                    raise RuntimeError("worker did not complete all fixture tokens")
                count = int(owned[0][4])
                if not 0 < count < requests["model_payload_bytes"]:
                    raise RuntimeError(f"rank {worker.rank} weight bytes {count}, reference {requests['model_payload_bytes']}")
                payload_bytes += count
                check_storage(worker, streamed, count)
                error = compare(data / f"oracle-{worker.rank}.f32le", worker.output,
                                128 if worker.rank == 2 else 512, worker.rank != 2)
                maximum = max(maximum, error)
            if payload_bytes != requests["model_payload_bytes"]:
                raise RuntimeError("stage weights do not cover the full target model exactly once")
            profile = "THREE_PROCESS_STREAMED_CPU_FIXTURE_PASS" if streamed else "THREE_PROCESS_CPU_FIXTURE_PASS"
            print(f"{profile}: A/B/C own 2/2/2 layers; 131 tokens; boundaries byte-exact; max logit error {maximum:.9g}")
            if streamed:
                print("STREAMED_SOURCE_EVIDENCE_PASS: A/C zero startup reads, deferred expert bytes, bounded actual inference I/O; B buffered; physical residency NOT_VALIDATED")
        finally:
            for worker in reversed(workers):
                worker.close()

        # With B absent, A must not authenticate directly to C or produce any inference output.
        rejected: list[Worker] = []
        try:
            c = Worker(executable, data, key_file, root / "missing-B-C.f32le", 2, streamed=streamed)
            rejected.append(c)
            a = Worker(executable, data, key_file, root / "missing-B-A.f32le", 0, c.listen_port(), streamed=streamed)
            rejected.append(a)
            for worker in rejected:
                worker.finish(False)
                if any(line.startswith("GLM_PIPELINE_DONE ") for line in worker.lines) or worker.output.stat().st_size:
                    raise RuntimeError("pipeline without B emitted model output")
            if "authentication" not in a.log_path.read_text(errors="replace").lower():
                raise RuntimeError("missing-middle case failed for an unrelated reason")
            print("MISSING_MIDDLE_REFUSAL_PASS: direct A-to-C startup failed before model output")
        finally:
            for worker in reversed(rejected):
                worker.close()

        # Same geometry and weights, different tokenizer metadata: peers must refuse before graph work.
        changed = root / "different tokenizer"
        changed.mkdir()
        shutil.copyfile(data / "model.gguf", changed / "model.gguf")
        shutil.copyfile(data / "requests.json", changed / "requests.json")
        with (changed / "model.gguf").open("r+b") as model_file:
            header = model_file.read(1 << 20)
            offset = header.find(b"tok_0")
            if offset < 0:
                raise RuntimeError("tokenizer identity fixture token was not found")
            model_file.seek(offset)
            model_file.write(b"tOk_0")
        mismatched: list[Worker] = []
        try:
            c = Worker(executable, data, key_file, root / "identity-C.f32le", 2, streamed=streamed)
            mismatched.append(c)
            b = Worker(executable, data, key_file, root / "identity-B.f32le", 1, c.listen_port(), streamed=streamed)
            mismatched.append(b)
            a = Worker(executable, changed, key_file, root / "identity-A.f32le", 0, b.listen_port(), streamed=streamed)
            mismatched.append(a)
            for worker in reversed(mismatched):
                worker.finish(False)
                if any(line.startswith("GLM_PIPELINE_DONE ") for line in worker.lines) or worker.output.stat().st_size:
                    raise RuntimeError("incompatible model identities produced inference output")
            if "negotiated identity mismatch" not in a.log_path.read_text(errors="replace").lower():
                raise RuntimeError("model identity mismatch failed for an unrelated reason")
            print("MODEL_IDENTITY_REFUSAL_PASS: changed tokenizer header refused before model output")
        finally:
            for worker in reversed(mismatched):
                worker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, type=pathlib.Path)
    parser.add_argument("--worker", required=True, type=pathlib.Path)
    args = parser.parse_args()
    run(args.oracle.resolve(strict=True), args.worker.resolve(strict=True))
    run(args.oracle.resolve(strict=True), args.worker.resolve(strict=True), streamed=True)
