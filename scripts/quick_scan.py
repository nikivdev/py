#!/usr/bin/env python3
"""Quick scan of known cache locations for immediate cleanup."""
import subprocess
import sys
from pathlib import Path

TARGETS = [
    ("Xcode DerivedData", "~/Library/Developer/Xcode/DerivedData"),
    ("Xcode Archives", "~/Library/Developer/Xcode/Archives"),
    ("Xcode Device Support", "~/Library/Developer/Xcode/iOS DeviceSupport"),
    ("iOS Simulators", "~/Library/Developer/CoreSimulator/Devices"),
    ("Xcode Cache", "~/Library/Caches/com.apple.dt.Xcode"),
    ("CocoaPods Cache", "~/Library/Caches/CocoaPods"),
    ("SPM Cache", "~/Library/Caches/org.swift.swiftpm"),
    ("npm Cache", "~/.npm"),
    ("Yarn Cache", "~/.yarn/cache"),
    ("pnpm Store", "~/Library/pnpm/store"),
    ("Bun Cache", "~/Library/Caches/bun"),
    ("pip Cache", "~/Library/Caches/pip"),
    ("uv Cache", "~/Library/Caches/uv"),
    ("Cargo Registry", "~/.cargo/registry"),
    ("Homebrew Cache", "~/Library/Caches/Homebrew"),
    ("Go Build Cache", "~/Library/Caches/go-build"),
    ("JetBrains Cache", "~/Library/Caches/JetBrains"),
    ("VSCode Cache", "~/Library/Application Support/Code/CachedData"),
    ("Cursor Cache", "~/Library/Application Support/Cursor/CachedData"),
    ("Spotify Cache", "~/Library/Caches/com.spotify.client"),
    ("Chrome Cache", "~/Library/Caches/Google"),
    ("Arc Cache", "~/Library/Caches/Arc"),
    ("Slack Cache", "~/Library/Application Support/Slack/Cache"),
    ("Discord Cache", "~/Library/Application Support/discord/Cache"),
    ("Electron Cache", "~/Library/Caches/electron"),
    ("Playwright Cache", "~/Library/Caches/ms-playwright"),
    ("Trash", "~/.Trash"),
    ("User Logs", "~/Library/Logs"),
    ("Docker", "~/Library/Containers/com.docker.docker/Data"),
    ("Ollama Models", "~/.ollama/models"),
    ("Huggingface Cache", "~/.cache/huggingface"),
    ("Torch Cache", "~/.cache/torch"),
]


def human_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def get_size(path: str) -> int:
    expanded = Path(path).expanduser()
    if not expanded.exists():
        return 0
    try:
        result = subprocess.run(
            ["du", "-sk", str(expanded)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return int(result.stdout.split()[0]) * 1024
    except:
        pass
    return 0


def main():
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--out", type=str, help="Write JSON to file")
    args = parser.parse_args()

    results = []
    total = 0

    if not args.json:
        print("Scanning cache locations...\n")

    for name, path in TARGETS:
        size = get_size(path)
        if size > 10 * 1024 * 1024:  # > 10MB
            total += size
            expanded = str(Path(path).expanduser())
            results.append({
                "name": name,
                "path": expanded,
                "size_bytes": size,
                "size_human": human_size(size),
                "category": "delete",
                "command": f'rm -rf "{expanded}"'
            })
            if not args.json:
                print(f"  {human_size(size):>10}  {name:<25}  {expanded}")

    results.sort(key=lambda x: x["size_bytes"], reverse=True)

    if args.json or args.out:
        output = {
            "total_bytes": total,
            "total_human": human_size(total),
            "items": results
        }
        if args.out:
            with open(args.out, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Wrote {args.out}")
        else:
            print(json.dumps(output, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Total reclaimable: {human_size(total)}")
        print(f"{'='*60}")

        print("\nTop 10 largest:")
        for item in results[:10]:
            print(f"\n  {item['name']} ({item['size_human']})")
            print(f"    {item['command']}")


if __name__ == "__main__":
    main()
