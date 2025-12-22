#!/usr/bin/env python3
"""Zip folders/files and (optionally) upload to S3 using aws cli."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def human_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def add_path(zf: zipfile.ZipFile, src: Path) -> None:
    if src.is_file():
        zf.write(src, src.name)
        return

    base = src.name
    for root, dirs, files in os.walk(src):
        dirs.sort()
        files.sort()
        rel_root = os.path.relpath(root, src)
        for name in files:
            full_path = Path(root) / name
            rel_path = Path(base) / rel_root / name if rel_root != "." else Path(base) / name
            zf.write(full_path, rel_path.as_posix())


def load_paths(paths: list[str], paths_file: str | None) -> list[Path]:
    collected: list[str] = []
    if paths_file:
        with open(paths_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    collected.append(line)
    collected.extend(paths)
    resolved = [Path(p).expanduser().resolve() for p in collected]
    return resolved


def path_hash(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    return digest[:8]


def zip_name_for(path: Path) -> str:
    safe_base = path.name.replace(" ", "_")
    return f"{safe_base}-{path_hash(path)}.zip"


def run_aws(args: list[str]) -> None:
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "aws command failed")


def s3_key(prefix: str, filename: str) -> str:
    return f"{prefix.rstrip('/')}/{filename}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zip files/directories and upload to S3.")
    parser.add_argument("paths", nargs="*", help="Files and directories to include")
    parser.add_argument("--paths-file", default=None, help="Read newline-separated paths from a file")
    parser.add_argument("--out-dir", default=None, help="Directory to write zip files (default: ./zip-staging)")
    parser.add_argument("--s3-prefix", default=None, help="S3 prefix like s3://bucket/path")
    parser.add_argument("--no-upload", action="store_true", help="Only create zip files, do not upload")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without creating zips or uploading")
    parser.add_argument("--overwrite", action="store_true", help="Recreate zips if they already exist")
    parser.add_argument("--aws-profile", default=None, help="AWS profile for aws cli")
    parser.add_argument("--manifest", default=None, help="Write JSONL manifest to this path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.paths and not args.paths_file:
        raise SystemExit("Provide paths or --paths-file")

    paths = load_paths(args.paths, args.paths_file)
    if not paths:
        raise SystemExit("No valid paths provided")

    out_dir = Path(args.out_dir or "zip-staging").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    upload = not args.no_upload
    if upload and not args.s3_prefix:
        raise SystemExit("Provide --s3-prefix for uploads or use --no-upload")

    manifest_path = args.manifest
    if manifest_path is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        manifest_path = str(out_dir / f"manifest-{ts}.jsonl")

    total_size = 0
    manifest_handle = open(manifest_path, "a", encoding="utf-8")
    try:
        for path in paths:
            if not path.exists():
                print(f"Skip missing: {path}", file=sys.stderr)
                continue

            zip_name = zip_name_for(path)
            zip_path = out_dir / zip_name
            s3_uri = s3_key(args.s3_prefix, zip_name) if args.s3_prefix else None

            action = "skip"
            error = None
            size_bytes = 0

            try:
                if zip_path.exists() and not args.overwrite:
                    action = "zip-skip"
                else:
                    if args.dry_run:
                        action = "zip-dry-run"
                    else:
                        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                            add_path(zf, path)
                        action = "zipped"

                if upload and action in {"zipped", "zip-skip"}:
                    if args.dry_run:
                        action = "upload-dry-run"
                    else:
                        aws_cmd = ["aws"]
                        if args.aws_profile:
                            aws_cmd.extend(["--profile", args.aws_profile])
                        aws_cmd.extend(["s3", "cp", str(zip_path), s3_uri])
                        run_aws(aws_cmd)
                        action = "uploaded"

                if zip_path.exists():
                    size_bytes = zip_path.stat().st_size
                    total_size += size_bytes
            except Exception as exc:
                action = "error"
                error = str(exc)

            record = {
                "source": str(path),
                "zip_path": str(zip_path),
                "zip_size": size_bytes,
                "zip_size_human": human_size(size_bytes),
                "s3_uri": s3_uri,
                "status": action,
                "error": error,
            }
            manifest_handle.write(json.dumps(record) + "\n")
            print(f"{action}: {path} -> {zip_path}")
    finally:
        manifest_handle.close()

    print(f"Wrote manifest: {manifest_path}")
    print(f"Total zip size: {human_size(total_size)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
