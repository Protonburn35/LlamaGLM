#!/usr/bin/env python3
"""Fetch only the audited GGUF headers, without model tensor payloads."""

import argparse
import hashlib
import pathlib
import urllib.request


REPOSITORY = "unsloth/GLM-5.3-Flash-GGUF"
REVISION = "a38483c8cd5df544f53d70fb281afe97369d5ab6"
# Header boundaries measured with the pinned llama.cpp GGUF reader. Shard 0 has no tensors.
HEADERS = [
    (1, 9429956, 9429984),
    (2, 38304, 49261511584),
    (3, 35232, 49075832480),
    (4, 15328, 22020797792),
]
HEADER_SHA256 = [
    "e47e6ba043a5d406cdf695219ff9728666dc3a0392eb7190171ca13fd671d63d",
    "ad6191c93269faeee16be8d2d1746df56f43e7f6515c26d646368a358f0ad7d4",
    "eed639822ed7b1e16efe6a4ab6d68728b5b3648c3270fe379218fb984097747e",
    "269b5ce802b9e424ad10d9781c709eace77e3310e5e1ba63939a3c074bfb20e3",
]


def fetch_header(number: int, header_bytes: int, file_bytes: int) -> bytes:
    name = f"GLM-5.3-Flash-UD-IQ3_XXS-{number:05d}-of-00004.gguf"
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/UD-IQ3_XXS/{name}"
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{header_bytes - 1}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        expected = f"bytes 0-{header_bytes - 1}/{file_bytes}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise RuntimeError("server did not honor the exact metadata range")
        if not response.geturl().startswith("https://"):
            raise RuntimeError("metadata redirect did not preserve HTTPS")
        data = response.read(header_bytes + 1)
    if len(data) != header_bytes or data[:4] != b"GGUF":
        raise RuntimeError("invalid or truncated metadata response")
    if hashlib.sha256(data).hexdigest() != HEADER_SHA256[number-1]:
        raise RuntimeError("metadata differs from the exact pinned header fingerprint")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=pathlib.Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for number, header_bytes, file_bytes in HEADERS:
        path = args.output_directory / f"GLM-5.3-Flash-UD-IQ3_XXS-{number:05d}-of-00004.gguf.header"
        if path.exists():
            raise FileExistsError("refusing to replace an existing metadata file")
        data = fetch_header(number, header_bytes, file_bytes)
        with path.open("xb") as stream:
            stream.write(data)
        print(f"Saved {path.name}: {header_bytes} metadata bytes")


if __name__ == "__main__":
    main()
