"""Pinned, checksum-verified downloads. No credentials, no automatic model updates."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for m in json.loads(Path(__file__).with_name("models.json").read_text()):
        target = args.directory / m["file"]
        if target.exists() and digest(target) == m["sha256"]:
            print(m["file"], "verified")
            continue
        partial = target.with_suffix(target.suffix + ".part")
        try:
            with urllib.request.urlopen(m["url"], timeout=120) as response, partial.open("wb") as f:
                while chunk := response.read(1024 * 1024):
                    f.write(chunk)
            if digest(partial) != m["sha256"]:
                raise RuntimeError("Model checksum mismatch: " + m["file"])
            partial.replace(target)
            print(m["file"], "downloaded and verified")
        finally:
            partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
