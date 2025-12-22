#!/usr/bin/env python3
"""Free space by deleting safe Xcode build artifacts and caches."""

import argparse
import shutil
from pathlib import Path


def human_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def get_dir_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean safe Xcode caches and build artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    parser.add_argument("--yes", action="store_true", help="Do not prompt for confirmation")
    parser.add_argument(
        "--include-device-support",
        action="store_true",
        help="Also delete iOS DeviceSupport (removes old device symbols)",
    )
    parser.add_argument(
        "--include-simulators",
        action="store_true",
        help="Also delete CoreSimulator devices (removes local simulators)",
    )
    parser.add_argument(
        "--include-archives",
        action="store_true",
        help="Also delete Xcode Archives (removes historic builds)",
    )
    args = parser.parse_args()

    targets: list[tuple[str, Path]] = [
        ("DerivedData", Path.home() / "Library/Developer/Xcode/DerivedData"),
        ("Xcode Caches", Path.home() / "Library/Caches/com.apple.dt.Xcode"),
    ]

    if args.include_device_support:
        targets.append(("iOS DeviceSupport", Path.home() / "Library/Developer/Xcode/iOS DeviceSupport"))

    if args.include_simulators:
        targets.append(("CoreSimulator Devices", Path.home() / "Library/Developer/CoreSimulator/Devices"))

    if args.include_archives:
        targets.append(("Xcode Archives", Path.home() / "Library/Developer/Xcode/Archives"))

    existing = [(name, path) for name, path in targets if path.exists()]
    if not existing:
        print("No Xcode cache/build directories found.")
        return

    print("Targets:")
    total_bytes = 0
    for name, path in existing:
        size = get_dir_size(path) if path.is_dir() else path.stat().st_size
        total_bytes += size
        print(f"  - {name}: {path} ({human_size(size)})")

    if args.dry_run:
        print(f"\nDry run: would delete {human_size(total_bytes)} total.")
        return

    if not args.yes:
        answer = input(f"\nDelete these items (~{human_size(total_bytes)})? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return

    deleted_bytes = 0
    for name, path in existing:
        size = get_dir_size(path) if path.is_dir() else path.stat().st_size
        try:
            remove_path(path)
            deleted_bytes += size
            print(f"Deleted {name}: {path} ({human_size(size)})")
        except OSError as exc:
            print(f"Failed to delete {name}: {path} ({exc})")

    print(f"\nFreed approximately {human_size(deleted_bytes)}.")


if __name__ == "__main__":
    main()
