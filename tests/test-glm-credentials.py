#!/usr/bin/env python3
"""Create public deterministic key fixtures with private/public native file policies."""

import argparse
import pathlib
import subprocess
import tempfile

from glm_private_files import secure_directory, make_public_test_file


def run(executable: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory(prefix="glm credential \u03bb ") as directory:
        root = pathlib.Path(directory)
        secure_directory(root)
        paths = [root / f"NEVER-ECHO-KEY-{i}.bin" for i in range(4)]
        for i,path in enumerate(paths):
            path.touch(mode=0o600,exist_ok=False)
            path.write_bytes(bytes(range(1,33)) if i < 2 else bytes(32 if i == 2 else 31))
        make_public_test_file(paths[1])
        result = subprocess.run([str(executable),*map(str,paths)],capture_output=True,text=True,timeout=30)
        if result.returncode or "six suites passed" not in result.stdout or "NEVER-ECHO" in result.stdout+result.stderr:
            raise RuntimeError("native credential lifecycle/privacy test failed")
        if paths[0].exists():
            raise RuntimeError("credential source was not deleted while native key pages remained active")
        print(result.stdout.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable",required=True,type=pathlib.Path)
    args = parser.parse_args()
    run(args.executable.resolve(strict=True))
