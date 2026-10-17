"""Prepare the dedicated bind-mount directory with numeric container ownership."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("data"))
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("Run with sudo for rootful Docker; see INSTALL.md for rootless UID mapping")
    p = args.directory
    if p.is_symlink():
        parser.error("Data directory must not be a symlink")
    p.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chown(p, 10001, 10001)
    p.chmod(0o700)
    print("Data directory prepared for container UID/GID 10001, mode 700")


if __name__ == "__main__":
    main()
