#!/usr/bin/env python3
"""Scan macOS for files/folders that can be deleted or moved to external storage."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

# Safe to delete (caches, build artifacts)
DELETE_TARGETS = [
    # Xcode / iOS Dev
    ("xcode_derived", Path.home() / "Library/Developer/Xcode/DerivedData"),
    ("xcode_archives", Path.home() / "Library/Developer/Xcode/Archives"),
    ("xcode_device_support", Path.home() / "Library/Developer/Xcode/iOS DeviceSupport"),
    ("ios_simulators", Path.home() / "Library/Developer/CoreSimulator/Devices"),
    ("xcode_cache", Path.home() / "Library/Caches/com.apple.dt.Xcode"),
    ("cocoapods_cache", Path.home() / "Library/Caches/CocoaPods"),
    ("spm_cache", Path.home() / "Library/Caches/org.swift.swiftpm"),
    # Package managers
    ("npm_cache", Path.home() / ".npm"),
    ("yarn_cache", Path.home() / ".yarn/cache"),
    ("pnpm_store", Path.home() / "Library/pnpm/store"),
    ("bun_cache", Path.home() / "Library/Caches/bun"),
    ("pip_cache", Path.home() / "Library/Caches/pip"),
    ("uv_cache", Path.home() / "Library/Caches/uv"),
    ("cargo_cache", Path.home() / ".cargo/registry"),
    ("homebrew_cache", Path.home() / "Library/Caches/Homebrew"),
    ("go_cache", Path.home() / "Library/Caches/go-build"),
    # IDEs
    ("jetbrains_cache", Path.home() / "Library/Caches/JetBrains"),
    ("vscode_cache", Path.home() / "Library/Application Support/Code/CachedData"),
    ("cursor_cache", Path.home() / "Library/Application Support/Cursor/CachedData"),
    # Apps
    ("spotify_cache", Path.home() / "Library/Caches/com.spotify.client"),
    ("chrome_cache", Path.home() / "Library/Caches/Google"),
    ("arc_cache", Path.home() / "Library/Caches/Arc"),
    ("slack_cache", Path.home() / "Library/Application Support/Slack/Cache"),
    ("discord_cache", Path.home() / "Library/Application Support/discord/Cache"),
    ("electron_cache", Path.home() / "Library/Caches/electron"),
    ("playwright_cache", Path.home() / "Library/Caches/ms-playwright"),
    ("rattler_cache", Path.home() / "Library/Caches/rattler"),
    # System
    ("trash", Path.home() / ".Trash"),
    ("user_logs", Path.home() / "Library/Logs"),
]

# Folders to scan for large files to move
MOVE_SCAN_PATHS = [
    Path.home() / "Desktop",
    Path.home() / "Documents",
    Path.home() / "Downloads",
    Path.home() / "Movies",
    Path.home() / "Pictures",
    Path.home() / "Music",
]

# Archive file extensions (good candidates for external storage)
ARCHIVE_EXTENSIONS = {
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".dmg", ".iso", ".pkg",
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".mp3", ".wav", ".flac", ".aac", ".m4a",
    ".pdf", ".epub",
    ".psd", ".ai", ".sketch",
    ".vmdk", ".vdi", ".qcow2",
}


@dataclass
class StorageItem:
    path: str
    size_bytes: int
    size_human: str
    category: Literal["delete", "move"]
    reason: str


def human_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def get_size_fast(path: Path) -> int:
    """Get folder size using du -s (fast)."""
    try:
        result = subprocess.run(
            ["du", "-sk", str(path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return int(result.stdout.split()[0]) * 1024
    except:
        pass
    return 0


def scan_delete_candidates() -> list[StorageItem]:
    items = []
    for name, path in DELETE_TARGETS:
        if path.exists():
            size = get_size_fast(path)
            if size > 50 * 1024 * 1024:  # > 50MB
                items.append(StorageItem(
                    path=str(path),
                    size_bytes=size,
                    size_human=human_size(size),
                    category="delete",
                    reason=name.replace("_", " ").title(),
                ))
    return items


def scan_node_modules() -> list[StorageItem]:
    """Find large node_modules folders."""
    items = []
    home = Path.home()
    try:
        result = subprocess.run(
            ["find", str(home), "-maxdepth", "3", "-type", "d", "-name", "node_modules"],
            capture_output=True,
            text=True,
            timeout=60
        )
        for line in result.stdout.strip().split('\n'):
            if line:
                path = Path(line)
                size = get_size_fast(path)
                if size > 100 * 1024 * 1024:  # > 100MB
                    items.append(StorageItem(
                        path=str(path),
                        size_bytes=size,
                        size_human=human_size(size),
                        category="delete",
                        reason="Node Modules",
                    ))
    except:
        pass
    return items


def scan_move_candidates(min_size_mb: int = 100) -> list[StorageItem]:
    """Scan for large files that can be moved to external storage."""
    items = []
    min_size = min_size_mb * 1024 * 1024

    for scan_path in MOVE_SCAN_PATHS:
        if not scan_path.exists():
            continue
        try:
            for entry in scan_path.rglob("*"):
                if entry.name.startswith(".") or "node_modules" in str(entry):
                    continue
                if not entry.is_file():
                    continue
                try:
                    size = entry.stat().st_size
                    if size >= min_size:
                        ext = entry.suffix.lower()
                        reason = f"Large file ({ext})" if ext in ARCHIVE_EXTENSIONS else "Large file"
                        items.append(StorageItem(
                            path=str(entry),
                            size_bytes=size,
                            size_human=human_size(size),
                            category="move",
                            reason=reason,
                        ))
                except:
                    pass
        except:
            pass
    return items


def main():
    print("Scanning for storage candidates...", file=sys.stderr)

    all_items: list[StorageItem] = []

    print("  Checking caches...", file=sys.stderr)
    all_items.extend(scan_delete_candidates())

    print("  Scanning for node_modules...", file=sys.stderr)
    all_items.extend(scan_node_modules())

    print("  Scanning for large files to move...", file=sys.stderr)
    all_items.extend(scan_move_candidates())

    # Sort by size
    all_items.sort(key=lambda x: x.size_bytes, reverse=True)

    # Dedupe
    seen = set()
    unique = []
    for item in all_items:
        if item.path not in seen:
            seen.add(item.path)
            unique.append(item)

    delete_items = [i for i in unique if i.category == "delete"]
    move_items = [i for i in unique if i.category == "move"]

    delete_total = sum(i.size_bytes for i in delete_items)
    move_total = sum(i.size_bytes for i in move_items)

    print(f"\n=== DELETE CANDIDATES ({human_size(delete_total)}) ===", file=sys.stderr)
    for item in delete_items[:20]:
        print(f"  {item.size_human:>10}  {item.reason:<20}  {item.path}", file=sys.stderr)

    print(f"\n=== MOVE CANDIDATES ({human_size(move_total)}) ===", file=sys.stderr)
    for item in move_items[:20]:
        print(f"  {item.size_human:>10}  {item.reason:<20}  {item.path}", file=sys.stderr)

    print(f"\nTotal potential savings: {human_size(delete_total + move_total)}", file=sys.stderr)

    # Output JSON
    output = {
        "scan_time": datetime.now().isoformat(),
        "summary": {
            "delete_count": len(delete_items),
            "delete_size": delete_total,
            "delete_size_human": human_size(delete_total),
            "move_count": len(move_items),
            "move_size": move_total,
            "move_size_human": human_size(move_total),
        },
        "items": [asdict(i) for i in unique],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
