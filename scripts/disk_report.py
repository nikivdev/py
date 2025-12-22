#!/usr/bin/env python3
"""Report disk usage for root volume."""

import shutil

def human_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def main() -> None:
    total, used, free = shutil.disk_usage("/")
    print("Disk usage for /")
    print(f"  total: {human_size(total)}")
    print(f"  used : {human_size(used)}")
    print(f"  free : {human_size(free)}")


if __name__ == "__main__":
    main()
