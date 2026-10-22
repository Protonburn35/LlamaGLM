"""Exercise the worker's reserve guard through actual unauthenticated RPC messages."""
import logging
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time


def receive(sock, size):
    data = bytearray()
    while len(data) < size:
        part = sock.recv(size - len(data))
        if not part:
            raise AssertionError("Worker closed connection")
        data.extend(part)
    return bytes(data)


def command(sock, cmd, payload):
    sock.sendall(struct.pack("<BQ", cmd, len(payload)) + payload)
    size, = struct.unpack("<Q", receive(sock, 8))
    assert size <= 64, size
    return receive(sock, size)


def check(server, reserve):
    with socket.socket() as temporary:
        temporary.bind(("127.0.0.1", 0))
        port = temporary.getsockname()[1]
    env = dict(os.environ, GGML_RPC_SERVER_RESERVE_MIB=reserve)
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen([server, "-d", "CPU", "-H", "127.0.0.1", "-p", str(port)],
                                   env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while True:
                if process.poll() is not None:
                    log.seek(0)
                    raise AssertionError(log.read().decode(errors="replace"))
                try:
                    sock = socket.create_connection(("127.0.0.1", port), timeout=1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.05)
            with sock:
                sock.settimeout(5)
                hello = command(sock, 14, bytes(24))
                assert len(hello) == 28, len(hello)
                # UINT64_MAX must be rejected before backend alignment/allocation.
                denied = command(sock, 0, struct.pack("<IQ", 0, (1 << 64) - 1))
                assert denied == bytes(16), (reserve, denied)
                small = command(sock, 0, struct.pack("<IQ", 0, 4096))
                ptr, size = struct.unpack("<QQ", small)
                if reserve == "1":
                    assert ptr and size >= 4096, (ptr, size)
                    sock.sendall(struct.pack("<BQ", 4, 8) + struct.pack("<Q", ptr))
                else:
                    assert not ptr and not size, (reserve, ptr, size)
                # The refused request must leave the worker usable.
                assert command(sock, 15, b"") == struct.pack("<I", 1)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    executable = str(Path(sys.argv[1]).resolve())
    for value in ("1", "-1", "garbage", "17592186044416", "18446744073709551616", "1,", "1,2", ""):
        check(executable, value)
    logging.info("PASS: 8 RPC reserve cases, overflow refusal, invalid-config refusal and continued service")
