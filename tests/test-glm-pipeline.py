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
                 rank: int, next_port: int = 0, *, streamed: bool = False, locked: bool = False,
                 session: str | None = None, generation: int = 1, controls: tuple[int, int] | None = None,
                 drop_ack: bool = False):
        self.rank = rank
        self.locked = locked
        self.source_removed = False
        self.control_port = 0
        self.model_path = data / "model.gguf"
        if locked and rank == 1:
            self.model_path = output.parent / "private-B" / output.stem / "model.gguf"
            self.model_path.parent.mkdir(parents=True)
            shutil.copyfile(data / "model.gguf", self.model_path)
        self.output = output
        self.log_path = output.with_suffix(".log")
        self.log = self.log_path.open("wb")
        command = [str(executable), "--rank", str(rank), "--model", str(self.model_path),
                   "--requests", str(data / "requests.json"), "--key-file", str(key_file), "--output", str(output),
                   "--storage", "locked" if locked else "streamed" if streamed else "buffered"]
        if next_port:
            command += ["--next-port", str(next_port)]
        if session is not None:
            command += ["--session", session, "--generation", str(generation)]
        if controls is not None:
            command += ["--control-b", str(controls[0]), "--control-c", str(controls[1])]
        if drop_ack:
            command += ["--drop-commit-ack", "1"]
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
            if fields[:1] == ["GLM_PIPELINE_CONTROL"]:
                if len(fields) != 3 or int(fields[1]) != self.rank or not 0 < int(fields[2]) < 65536 or self.control_port:
                    raise RuntimeError("invalid fixture control listener announcement")
                self.control_port = int(fields[2])
            if fields[:1] == ["GLM_PIPELINE_LISTEN"]:
                if len(fields) != 3 or int(fields[1]) != self.rank or not 0 < int(fields[2]) < 65536:
                    raise RuntimeError("invalid fixture listener announcement")
                if self.locked and self.rank == 1:
                    self.model_path.unlink()
                    self.source_removed = True
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


def check_storage(worker: Worker, streamed: bool, logical_bytes: int, locked: bool) -> None:
    records = [json.loads(line.removeprefix("GLM_PIPELINE_STORAGE ")) for line in worker.lines
               if line.startswith("GLM_PIPELINE_STORAGE ")]
    if len(records) != 2 or [record["phase"] for record in records] != ["startup", "complete"]:
        raise RuntimeError("worker storage evidence is missing or out of order")
    desktop = streamed and worker.rank != 1
    preloaded = locked and worker.rank == 1
    has_source = desktop or preloaded
    layers = ([1, 1, 3] if locked else [1, 2, 2])[worker.rank]
    banks = [json.loads(line.removeprefix("GLM_PIPELINE_BANK ")) for line in worker.lines
             if line.startswith("GLM_PIPELINE_BANK ")]
    if len(banks) != (1 if has_source else 0):
        raise RuntimeError("worker bank reuse evidence is missing or has the wrong provider")
    if has_source:
        bank = banks[0]
        if not (bank["rank"] == worker.rank and bank["cpu_banks"] == layers and bank["device_banks"] == 0 and
                0 < bank["buffer_bytes"] <= 16 << 20 and bank["materializations"] == layers * 131 and
                bank["hits"] > 0 and bank["misses"] > 0 and bank["hits"] + bank["misses"] == layers * 131 * 8 and
                bank["copied_slabs"] == bank["misses"] * 3 and bank["copied_bytes"] > 0 and bank["failures"] == 0):
            raise RuntimeError("worker backend slot/copy counters do not prove complete hot expert reuse")
        print(f"BACKEND_BANK_RANK {worker.rank}: CPU hits {bank['hits']}; misses {bank['misses']}; "
              f"copied slabs {bank['copied_slabs']}; copied bytes {bank['copied_bytes']}; device execution NOT_RUN")
    for record in records:
        if (record["version"], record["rank"], record["microbatch"], record["mode"]) != (
                1, worker.rank, 1 if streamed else 32, "desktop-streamed" if desktop else "host-preloaded" if preloaded else "buffered"):
            raise RuntimeError("worker executed the wrong storage/microbatch profile")
        if has_source:
            if not (0 < record["routed_logical_bytes"] < logical_bytes and
                    0 < record["model_buffer_bytes"] < logical_bytes and
                    record["streamed_tensors"] == 3 * layers and
                    0 <= record["host_reserved_bytes"] <= 16 << 20):
                raise RuntimeError("streamed worker did not defer experts or exceeded its host payload cap")
            if preloaded:
                if not (record["expert_slabs"] == record["native_read_requests"] == 48 and
                        record["native_read_bytes"] == record["host_reserved_bytes"] == record["routed_logical_bytes"] and
                        0 < record["routed_logical_bytes"] <= record["locked_expert_bytes"] <= 2 << 20 and
                        record["locked_expert_bytes"] < record["host_page_bytes"] <= record["peak_host_page_bytes"] <= 8 << 20 and
                        record["host_buffers"] > 0 and record["native_locks_verified"] == 1 and
                        record["full_residency_validated"] == record["open_source_files"] == 0 and
                        record["source_removed"] == (record["phase"] == "complete")):
                    raise RuntimeError("middle worker lost its complete locked expert/host payload coverage")
            elif record["phase"] == "startup":
                if any(record[key] != 0 for key in ("native_read_requests", "native_read_bytes", "host_reserved_bytes")):
                    raise RuntimeError("streamed worker read expert payloads during startup")
            elif record["native_read_requests"] <= 0 or record["native_read_bytes"] <= 0:
                raise RuntimeError("streamed model worker performed no source reads")
        elif any(key.startswith("native_read") for key in record):
            raise RuntimeError("buffered worker unexpectedly exposed a desktop disk source")
    if has_source and any(records[0][key] != records[1][key] for key in
                       ("routed_logical_bytes", "model_buffer_bytes", "streamed_tensors")):
        raise RuntimeError("streamed worker changed weight ownership during inference")
    if preloaded:
        if not (worker.source_removed and not worker.model_path.exists() and
                records[1]["host_page_bytes"] > records[0]["host_page_bytes"] and
                records[1]["host_buffers"] > records[0]["host_buffers"] and
                all(records[0][key] == records[1][key] for key in
                    ("native_read_requests", "native_read_bytes", "host_reserved_bytes", "locked_expert_bytes"))):
            raise RuntimeError("middle worker reused its source file or omitted context payload protection")
        print(f"LOCKED_HOST_RANK 1: all 48 expert slabs; source removed; shared peak {records[1]['peak_host_page_bytes']}; "
              "tracked native locks verified; full working-set admission NOT_VALIDATED")
    if desktop:
        final = records[1]
        print(f"STREAMED_SOURCE_RANK {worker.rank}: routed logical {final['routed_logical_bytes']}; "
              f"ordinary buffers {final['model_buffer_bytes']}; host reserved {final['host_reserved_bytes']}; "
              f"native reads {final['native_read_requests']}; native requested bytes {final['native_read_bytes']}")


def run(oracle: pathlib.Path, executable: pathlib.Path, *, streamed: bool = False, locked: bool = False) -> None:
    streamed = streamed or locked
    with tempfile.TemporaryDirectory(prefix="glm three process fixture ") as directory:
        root = pathlib.Path(directory)
        data = root / "oracle and model"
        command = [str(oracle), "--glm-pipeline-fixture", str(data), "--seed", "53"]
        if locked:
            command += ["--glm-pipeline-locked"]
        elif streamed:
            command += ["--glm-pipeline-streamed"]
        generation = subprocess.run(command,
                                    capture_output=True, text=True, timeout=60)
        if generation.returncode != 0 or "GLM pipeline oracle:" not in generation.stdout:
            raise RuntimeError(f"oracle generation failed: {generation.stdout[-2000:]} {generation.stderr[-4000:]}")
        requests = json.loads((data / "requests.json").read_text())
        if requests["microbatch"] != (1 if streamed else 32):
            raise RuntimeError("independent monolithic oracle used the wrong microbatch")
        ends = [2, 3, 6] if locked else [2, 4, 6]
        if requests["layer_ends"] != ends:
            raise RuntimeError("independent monolithic oracle used the wrong boundaries")
        key_file = root / "ephemeral-fixture-key.bin"
        key_file.touch(mode=0o600, exist_ok=False)
        key_file.write_bytes(secrets.token_bytes(32))
        workers: list[Worker] = []
        try:
            c = Worker(executable, data, key_file, root / "stage-C.f32le", 2, streamed=streamed, locked=locked)
            workers.append(c)
            b = Worker(executable, data, key_file, root / "stage-B.f32le", 1, c.listen_port(), streamed=streamed, locked=locked)
            workers.append(b)
            a = Worker(executable, data, key_file, root / "stage-A.f32le", 0, b.listen_port(), streamed=streamed, locked=locked)
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
                if len(owned) != 1 or len(owned[0]) != 5 or list(map(int, owned[0][1:4])) != [
                        worker.rank, ends[worker.rank - 1] if worker.rank else 0, ends[worker.rank]]:
                    raise RuntimeError("worker did not report its exact owned layer range")
                if len(done) != 1 or list(map(int, done[0][1:])) != [worker.rank, 131]:
                    raise RuntimeError("worker did not complete all fixture tokens")
                count = int(owned[0][4])
                if not 0 < count < requests["model_payload_bytes"]:
                    raise RuntimeError(f"rank {worker.rank} weight bytes {count}, reference {requests['model_payload_bytes']}")
                payload_bytes += count
                check_storage(worker, streamed, count, locked)
                error = compare(data / f"oracle-{worker.rank}.f32le", worker.output,
                                128 if worker.rank == 2 else 512, worker.rank != 2)
                maximum = max(maximum, error)
            if payload_bytes != requests["model_payload_bytes"]:
                raise RuntimeError("stage weights do not cover the full target model exactly once")
            profile = "THREE_PROCESS_LOCKED_HOST_CPU_FIXTURE_PASS" if locked else "THREE_PROCESS_STREAMED_CPU_FIXTURE_PASS" if streamed else "THREE_PROCESS_CPU_FIXTURE_PASS"
            split = "2/1/3" if locked else "2/2/2"
            print(f"{profile}: A/B/C own {split} layers; 131 tokens; boundaries byte-exact; max logit error {maximum:.9g}")
            if streamed and not locked:
                print("STREAMED_SOURCE_EVIDENCE_PASS: A/C zero startup reads, deferred expert bytes, bounded actual inference I/O; B buffered; physical residency NOT_VALIDATED")
        finally:
            for worker in reversed(workers):
                worker.close()

        # With B absent, A must not authenticate directly to C or produce any inference output.
        rejected: list[Worker] = []
        try:
            c = Worker(executable, data, key_file, root / "missing-B-C.f32le", 2, streamed=streamed, locked=locked)
            rejected.append(c)
            a = Worker(executable, data, key_file, root / "missing-B-A.f32le", 0, c.listen_port(), streamed=streamed, locked=locked)
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
            c = Worker(executable, data, key_file, root / "identity-C.f32le", 2, streamed=streamed, locked=locked)
            mismatched.append(c)
            b = Worker(executable, data, key_file, root / "identity-B.f32le", 1, c.listen_port(), streamed=streamed, locked=locked)
            mismatched.append(b)
            a = Worker(executable, changed, key_file, root / "identity-A.f32le", 0, b.listen_port(), streamed=streamed, locked=locked)
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


def run_transactions(oracle: pathlib.Path, executable: pathlib.Path) -> None:
    """Exercise real graph commits, lost B ACK and process-restart replay with fresh keys."""
    with tempfile.TemporaryDirectory(prefix="glm transaction fixture ") as directory:
        root = pathlib.Path(directory)
        data = root / "independent oracle"
        generated = subprocess.run([str(oracle), "--glm-pipeline-fixture", str(data), "--glm-pipeline-locked", "--seed", "53"],
                                   capture_output=True, text=True, timeout=60)
        if generated.returncode != 0 or "GLM pipeline oracle:" not in generated.stdout:
            raise RuntimeError(f"transaction oracle failed: {generated.stderr[-4000:]}")
        requests = json.loads((data / "requests.json").read_text())

        def attempt(label: str, generation: int, drop_ack: bool) -> None:
            # New credential and session for each process group; never resume old connections.
            key_file = root / f"{label}-key.bin"
            key_file.touch(mode=0o600, exist_ok=False)
            key_file.write_bytes(secrets.token_bytes(32))
            session = secrets.token_hex(16)
            workers: list[Worker] = []
            try:
                c = Worker(executable, data, key_file, root / f"{label}-C.f32le", 2,
                           locked=True, session=session, generation=generation)
                workers.append(c)
                c_port = c.listen_port()
                b = Worker(executable, data, key_file, root / f"{label}-B.f32le", 1, c_port,
                           locked=True, session=session, generation=generation, drop_ack=drop_ack)
                workers.append(b)
                b_port = b.listen_port()
                if not b.control_port or not c.control_port:
                    raise RuntimeError("transaction fixture omitted independent control listeners")
                a = Worker(executable, data, key_file, root / f"{label}-A.f32le", 0, b_port, locked=True,
                           session=session, generation=generation, controls=(b.control_port, c.control_port))
                workers.append(a)
                if len({worker.process.pid for worker in workers}) != 3:
                    raise RuntimeError("transaction fixture did not create three distinct model processes")
                for worker in reversed(workers):
                    worker.finish(not drop_ack)
                if drop_ack:
                    for worker in workers:
                        if worker.output.stat().st_size or f"GLM_PIPELINE_INVALIDATED {worker.rank}" not in worker.lines:
                            raise RuntimeError("uncertain commit emitted output or failed to invalidate every real owner")
                        if any(line.startswith(("GLM_PIPELINE_DONE ", "GLM_PIPELINE_TRANSACTIONS ")) for line in worker.lines):
                            raise RuntimeError("uncertain commit was reported complete")
                    if "injected lost B commit acknowledgement after local commit" not in b.log_path.read_text(errors="replace"):
                        raise RuntimeError("lost-ACK test failed before the intended partial commit")
                    print("LOST_COMMIT_ACK_REFUSAL_PASS: B committed then dropped ACK; A/B/C invalidated; zero output")
                    return
                certificates: list[list[str]] = []
                total_owned = 0
                maximum = 0.0
                for worker in workers:
                    records = [json.loads(line.removeprefix("GLM_PIPELINE_TRANSACTIONS ")) for line in worker.lines
                               if line.startswith("GLM_PIPELINE_TRANSACTIONS ")]
                    if len(records) != 1:
                        raise RuntimeError("worker did not report authenticated transaction completion")
                    record = records[0]
                    if not (record["rank"] == worker.rank and record["generation"] == generation and
                            record["executions"] == record["committed_retries"] == 7 and record["committed_position"] == 131 and
                            0 < record["reply_bytes"] <= record["peak_reply_bytes"] <= 256 << 10 and
                            (0 < record["reply_page_bytes"] <= record["peak_reply_page_bytes"] <= 256 << 10 if worker.rank == 1 else
                             record["reply_page_bytes"] == record["peak_reply_page_bytes"] == 0) and
                            record["full_residency_validated"] is False and len(record["certificates"]) == 7 and
                            len(set(record["certificates"])) == 7):
                        raise RuntimeError("worker transaction cursor/retry/certificate evidence differs")
                    certificates.append(record["certificates"])
                    print(f"REPLY_CACHE_RANK {worker.rank}: payload peak {record['peak_reply_bytes']}; "
                          f"native page peak {record['peak_reply_page_bytes']}; seven immutable committed retries")
                    owned = [line.split() for line in worker.lines if line.startswith("GLM_PIPELINE_OWNED ")]
                    ends = [2, 3, 6]
                    if len(owned) != 1 or list(map(int, owned[0][1:4])) != [worker.rank, ends[worker.rank-1] if worker.rank else 0, ends[worker.rank]]:
                        raise RuntimeError("transaction worker changed its assigned model layers")
                    owned_bytes = int(owned[0][4]); total_owned += owned_bytes
                    check_storage(worker, True, owned_bytes, True)
                    maximum = max(maximum, compare(data / f"oracle-{worker.rank}.f32le", worker.output,
                                                   128 if worker.rank == 2 else 512, True))
                if total_owned != requests["model_payload_bytes"] or not all(values == certificates[0] for values in certificates):
                    raise RuntimeError("distributed owners disagree on weight coverage or complete commit certificates")
                print(f"THREE_PROCESS_TRANSACTIONS_PASS: {label}; seven all-owner commits/retries; 131 tokens; "
                      f"byte-exact boundaries/logits; max error {maximum:.9g}; full admission NOT_VALIDATED")
            finally:
                for worker in reversed(workers):
                    worker.close()
                key_file.unlink()

        attempt("committed", 1, False)
        attempt("lost-ack", 1, True)
        attempt("process-restart-replay", 2, False)
        print("PROCESS_RESTART_REPLAY_PASS: fresh credential/session/generation, replay from zero; retained-weight distributed reset NOT_IMPLEMENTED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, type=pathlib.Path)
    parser.add_argument("--worker", required=True, type=pathlib.Path)
    args = parser.parse_args()
    run(args.oracle.resolve(strict=True), args.worker.resolve(strict=True))
    run(args.oracle.resolve(strict=True), args.worker.resolve(strict=True), streamed=True)
    run(args.oracle.resolve(strict=True), args.worker.resolve(strict=True), locked=True)
    run_transactions(args.oracle.resolve(strict=True), args.worker.resolve(strict=True))
