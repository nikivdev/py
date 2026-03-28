#!/usr/bin/env python3
"""
Visualize file system concepts: inodes, directory structure, file descriptors.
"""

import os
import stat
import sys
import pwd
import grp
from datetime import datetime

# ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"


def visualize_inode_structure():
    """Show inode structure and how it maps to data blocks."""
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}              INODE STRUCTURE{RESET}")
    print(f"{BOLD}{'═' * 65}{RESET}\n")

    print(f"  {DIM}An inode stores metadata about a file (but NOT the filename!){RESET}")
    print()

    # Inode structure
    print(f"  {BOLD}Inode Contents:{RESET}")
    print("  ┌────────────────────────────────────────────────────────┐")
    print(f"  │  {CYAN}File Type & Permissions{RESET}   -rw-r--r--                │")
    print(f"  │  {CYAN}Owner UID{RESET}                 501                       │")
    print(f"  │  {CYAN}Group GID{RESET}                 20                        │")
    print(f"  │  {CYAN}File Size{RESET}                 4096 bytes                │")
    print(f"  │  {CYAN}Timestamps{RESET}                                          │")
    print("  │    - atime (last access)  2024-01-15 10:30:00         │")
    print("  │    - mtime (last modify)  2024-01-14 09:15:00         │")
    print("  │    - ctime (inode change) 2024-01-14 09:15:00         │")
    print(f"  │  {CYAN}Link Count{RESET}                2                         │")
    print(f"  │  {CYAN}Data Block Pointers{RESET}                                 │")
    print("  │    - Direct blocks [0-11]                             │")
    print("  │    - Single indirect                                  │")
    print("  │    - Double indirect                                  │")
    print("  │    - Triple indirect                                  │")
    print("  └────────────────────────────────────────────────────────┘")
    print()

    # Block pointer visualization
    print(f"  {BOLD}Data Block Pointers:{RESET}")
    print("""
    Inode                    Data Blocks
    ┌─────────────┐         ┌────────┐
    │ Direct 0   ─┼────────►│Block 42│
    │ Direct 1   ─┼────────►│Block 87│
    │ Direct 2   ─┼────────►│Block 23│
    │   ...       │         │  ...   │
    │ Direct 11   │         └────────┘
    ├─────────────┤
    │ Single     ─┼──┐      Indirect Block
    │ Indirect    │  │     ┌─────────────┐
    ├─────────────┤  └────►│ ptr → Blk 5 │
    │ Double     ─┼─┐      │ ptr → Blk 9 │
    │ Indirect    │ │      │ ptr → Blk 12│
    ├─────────────┤ │      │    ...      │
    │ Triple      │ │      └─────────────┘
    │ Indirect    │ │
    └─────────────┘ │       Double Indirect
                    │      ┌──────────────┐
                    └─────►│ ptr → IndBlk │──► more blocks
                           │ ptr → IndBlk │──► more blocks
                           └──────────────┘
    """)


def visualize_directory_structure():
    """Show how directories work (directory = list of name → inode mappings)."""
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}           DIRECTORY STRUCTURE{RESET}")
    print(f"{BOLD}{'═' * 65}{RESET}\n")

    print(f"  {DIM}A directory is just a special file containing name → inode mappings{RESET}")
    print()

    print(f"  {BOLD}Example: /home/user directory{RESET}")
    print("  ┌────────────────────────────────────────────┐")
    print("  │ Directory Entry (dirent)                  │")
    print("  ├──────────────────┬─────────────────────────┤")
    print("  │   Filename       │   Inode Number          │")
    print("  ├──────────────────┼─────────────────────────┤")
    print(f"  │   {DIM}.{RESET}              │   {CYAN}1001{RESET}  (self)           │")
    print(f"  │   {DIM}..{RESET}             │   {CYAN}1000{RESET}  (parent: /home)  │")
    print(f"  │   {GREEN}documents{RESET}      │   {CYAN}1002{RESET}  (directory)      │")
    print(f"  │   {YELLOW}file.txt{RESET}       │   {CYAN}1003{RESET}  (regular file)   │")
    print(f"  │   {MAGENTA}link.txt{RESET}       │   {CYAN}1003{RESET}  (hard link!)     │")
    print(f"  │   {BLUE}shortcut{RESET}       │   {CYAN}1004{RESET}  (symlink)        │")
    print("  └──────────────────┴─────────────────────────┘")
    print()

    # Hard link vs symlink
    print(f"  {BOLD}Hard Link vs Symbolic Link:{RESET}")
    print(f"""
    {YELLOW}Hard Link:{RESET}                    {BLUE}Symbolic Link:{RESET}
    file.txt ──┐                  shortcut ──► "/home/user/file.txt"
               │                                    │
    link.txt ──┴──► Inode 1003 ──► Data       (stores path as data)
                         │                          │
                    Link count: 2              Points to → file.txt
    """)


def show_file_info(path):
    """Show detailed file information (like stat command)."""
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}         FILE INFORMATION: {path}{RESET}")
    print(f"{BOLD}{'═' * 65}{RESET}\n")

    try:
        st = os.stat(path)
        lst = os.lstat(path)  # Don't follow symlinks
    except FileNotFoundError:
        print(f"  {RED}File not found: {path}{RESET}")
        return

    # File type
    if stat.S_ISREG(st.st_mode):
        ftype = "Regular File"
    elif stat.S_ISDIR(st.st_mode):
        ftype = "Directory"
    elif stat.S_ISLNK(lst.st_mode):
        ftype = f"Symbolic Link → {os.readlink(path)}"
    elif stat.S_ISBLK(st.st_mode):
        ftype = "Block Device"
    elif stat.S_ISCHR(st.st_mode):
        ftype = "Character Device"
    elif stat.S_ISFIFO(st.st_mode):
        ftype = "FIFO/Pipe"
    elif stat.S_ISSOCK(st.st_mode):
        ftype = "Socket"
    else:
        ftype = "Unknown"

    # Permissions
    perms = stat.filemode(st.st_mode)

    # Owner/Group
    try:
        owner = pwd.getpwuid(st.st_uid).pw_name
    except KeyError:
        owner = str(st.st_uid)
    try:
        group = grp.getgrgid(st.st_gid).gr_name
    except KeyError:
        group = str(st.st_gid)

    print(f"  ┌{'─' * 50}┐")
    print(f"  │ {CYAN}Type:{RESET}        {ftype:<37}│")
    print(f"  │ {CYAN}Inode:{RESET}       {st.st_ino:<37}│")
    print(f"  │ {CYAN}Permissions:{RESET} {perms:<37}│")
    print(f"  │ {CYAN}Owner:{RESET}       {owner} (UID: {st.st_uid}){' ' * (25 - len(owner) - len(str(st.st_uid)))}│")
    print(f"  │ {CYAN}Group:{RESET}       {group} (GID: {st.st_gid}){' ' * (25 - len(group) - len(str(st.st_gid)))}│")
    print(f"  │ {CYAN}Size:{RESET}        {st.st_size} bytes{' ' * (30 - len(str(st.st_size)))}│")
    print(f"  │ {CYAN}Hard Links:{RESET}  {st.st_nlink:<37}│")
    print(f"  │ {CYAN}Device:{RESET}      {st.st_dev:<37}│")
    print(f"  ├{'─' * 50}┤")
    print(f"  │ {YELLOW}Access Time:{RESET} {datetime.fromtimestamp(st.st_atime).strftime('%Y-%m-%d %H:%M:%S'):<26}│")
    print(f"  │ {YELLOW}Modify Time:{RESET} {datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S'):<26}│")
    print(f"  │ {YELLOW}Change Time:{RESET} {datetime.fromtimestamp(st.st_ctime).strftime('%Y-%m-%d %H:%M:%S'):<26}│")
    print(f"  └{'─' * 50}┘")


def visualize_file_descriptors():
    """Show file descriptor table and how it works."""
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}           FILE DESCRIPTORS{RESET}")
    print(f"{BOLD}{'═' * 65}{RESET}\n")

    print(f"  {DIM}Each process has a file descriptor table{RESET}")
    print()

    print(f"  {BOLD}Process File Descriptor Table:{RESET}")
    print(f"""
    Process A                     System-wide Tables
    ┌─────────────────┐          ┌─────────────────────┐
    │ FD Table        │          │ Open File Table     │
    ├─────┬───────────┤          ├─────────────────────┤
    │  0  │ stdin    ─┼────┐     │ Entry 0             │
    │  1  │ stdout   ─┼────┼────►│  - offset: 0        │
    │  2  │ stderr   ─┼────┤     │  - flags: O_RDONLY  │──► Inode
    │  3  │ file.txt ─┼────┼────►│  - ref count: 2     │    Table
    │  4  │ socket   ─┼────┼────►│ Entry 1             │     │
    │  5  │ (unused)  │    │     │  - offset: 1024     │     ▼
    └─────┴───────────┘    │     │  - flags: O_RDWR    │   ┌──────┐
                           │     └─────────────────────┘   │Inode │
    Process B              │                               │ 1003 │
    ┌─────────────────┐    │     {DIM}(shared between{RESET}          │      │
    │ FD Table        │    │      {DIM}fork()ed processes){RESET}     └──────┘
    ├─────┬───────────┤    │
    │  3  │ file.txt ─┼────┘     {DIM}Offset is per open-file,{RESET}
    └─────┴───────────┘          {DIM}not per file descriptor!{RESET}
    """)

    # Show actual FDs for this process
    print(f"  {BOLD}Current Process File Descriptors:{RESET}")
    print("  ┌──────┬────────────────────────────────────────────────┐")
    for fd in range(10):
        try:
            path = os.readlink(f"/dev/fd/{fd}")
            print(f"  │  {fd}   │ {path:<47}│")
        except (OSError, FileNotFoundError):
            pass
    print("  └──────┴────────────────────────────────────────────────┘")


def visualize_disk_layout():
    """Show typical disk/partition layout."""
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}           DISK LAYOUT (ext4-like){RESET}")
    print(f"{BOLD}{'═' * 65}{RESET}\n")

    print(f"""
    Disk Layout
    ┌─────────────────────────────────────────────────────────────────┐
    │{YELLOW}Boot{RESET}│{CYAN}  Superblock  {RESET}│{GREEN}Block Group 0{RESET}│{GREEN}Block Group 1{RESET}│{GREEN}  ...  {RESET}│{GREEN}Group N{RESET}│
    │{YELLOW}Sect{RESET}│{CYAN}  (metadata) {RESET}│             │             │       │       │
    └─────────────────────────────────────────────────────────────────┘
       │        │
       │        └── Contains: filesystem size, block size,
       │            free blocks count, free inodes count,
       │            mount count, magic number...
       │
       └── Boot code (if bootable)

    Block Group Structure
    ┌───────────────────────────────────────────────────────────────────┐
    │{CYAN}Group{RESET}│{MAGENTA}Block{RESET} │{MAGENTA}Inode{RESET} │{YELLOW}Inode{RESET}   │{GREEN}Data Blocks{RESET}                      │
    │{CYAN}Desc {RESET}│{MAGENTA}Bitmap{RESET}│{MAGENTA}Bitmap{RESET}│{YELLOW}Table{RESET}   │{GREEN}                                  {RESET}│
    └───────────────────────────────────────────────────────────────────┘
       │       │       │       │            │
       │       │       │       │            └── Actual file data
       │       │       │       │
       │       │       │       └── Array of inodes (metadata)
       │       │       │
       │       │       └── Tracks which inodes are free/used
       │       │
       │       └── Tracks which blocks are free/used
       │
       └── Describes this block group

    {DIM}Example: Finding a file{RESET}
    1. Read root inode (always inode 2)
    2. Read root's data blocks (directory entries)
    3. Find filename → inode number mapping
    4. Read that inode to get data block pointers
    5. Read data blocks
    """)


def interactive_menu():
    """Interactive menu."""
    while True:
        print(f"\n{BOLD}{'═' * 65}{RESET}")
        print(f"{BOLD}     FILE SYSTEM EXPLORER{RESET}")
        print(f"{BOLD}{'═' * 65}{RESET}")
        print(f"""
  {CYAN}1.{RESET} Inode structure
  {CYAN}2.{RESET} Directory structure
  {CYAN}3.{RESET} File descriptors
  {CYAN}4.{RESET} Disk layout
  {CYAN}5.{RESET} Inspect a file (stat)
  {CYAN}q.{RESET} Quit
        """)

        choice = input(f"{YELLOW}Choose option: {RESET}").strip().lower()

        if choice == "1":
            visualize_inode_structure()
        elif choice == "2":
            visualize_directory_structure()
        elif choice == "3":
            visualize_file_descriptors()
        elif choice == "4":
            visualize_disk_layout()
        elif choice == "5":
            path = input("  Enter file path (or press Enter for this script): ").strip()
            if not path:
                path = __file__
            show_file_info(path)
        elif choice == "q":
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        visualize_inode_structure()
        visualize_directory_structure()
    else:
        interactive_menu()
