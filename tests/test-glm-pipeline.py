#!/usr/bin/env python3
"""Qualify three actual owned-layer CPU processes against a separate monolithic oracle."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import queue
import secrets
import struct
import subprocess
import tempfile
import threading


class Worker:
    def __init__(self, executable: pathlib.Path, data: pathlib.Path, key_file: pathlib.Path, output: pathlib.Path, rank: int, next_port: int = 0):
        self.rank = rank
        self.output = output
        self.log_path = output.with_suffix(".log")
        self.log = self.log_path.open("wb")
        command = [str(executable), "--rank", str(rank), "--model", str(data / "model.gguf"),
                   "--requests", str(data / "requests.json"), "--key-file", str(key_file), "--output", str(output)]
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


def run(oracle: pathlib.Path, executable: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory(prefix="glm three process fixture ") as directory:
        root = pathlib.Path(directory)
        data = root / "oracle and model"
        generation = subprocess.run([str(oracle), "--glm-pipeline-fixture", str(data), "--seed", "53"],
                                    capture_output=True, text=True, timeout=60)
        if generation.returncode != 0 or "GLM pipeline oracle:" not in generation.stdout:
            raise RuntimeError(f"oracle generation failed: {generation.stdout[-2000:]} {generation.stderr[-4000:]}")
        requests = json.loads((data / "requests.json").read_text())
        key_file = root / "ephemeral-fixture-key.bin"
        key_file.touch(mode=0o600, exist_ok=False)
        key_file.write_bytes(secrets.token_bytes(32))
        workers: list[Worker] = []
        try:
            c = Worker(executable, data, key_file, root / "stage-C.f32le", 2)
            workers.append(c)
            b = Worker(executable, data, key_file, root / "stage-B.f32le", 1, c.listen_port())
            workers.append(b)
            a = Worker(executable, data, key_file, root / "stage-A.f32le", 0, b.listen_port())
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
                error = compare(data / f"oracle-{worker.rank}.f32le", worker.output,
                                128 if worker.rank == 2 else 512, worker.rank != 2)
                maximum = max(maximum, error)
            if payload_bytes != requests["model_payload_bytes"]:
                raise RuntimeError("stage weights do not cover the full target model exactly once")
            print(f"THREE_PROCESS_CPU_FIXTURE_PASS: A/B/C own 2/2/2 layers; 131 tokens; boundaries byte-exact; max logit error {maximum:.9g}")
        finally:
            for worker in reversed(workers):
                worker.close()

        # With B absent, A must not authenticate directly to C or produce any inference output.
        rejected: list[Worker] = []
        try:
            c = Worker(executable, data, key_file, root / "missing-B-C.f32le", 2)
            rejected.append(c)
            a = Worker(executable, data, key_file, root / "missing-B-A.f32le", 0, c.listen_port())
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, type=pathlib.Path)
    parser.add_argument("--worker", required=True, type=pathlib.Path)
    args = parser.parse_args()
    run(args.oracle.resolve(strict=True), args.worker.resolve(strict=True))
