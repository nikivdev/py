#!/usr/bin/env python3
"""
Operating System Concepts Explorer
Interactive learning tool for understanding OS fundamentals.
"""

import os
import sys
import subprocess

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

MODULES = {
    "1": ("process_memory", "Process Memory Layout", "Stack, heap, data, text sections"),
    "2": ("process_fork", "Process Creation (Fork)", "Fork, copy-on-write, process tree"),
    "3": ("scheduler", "CPU Scheduling", "FCFS, SJF, Round Robin, Priority"),
    "4": ("virtual_memory", "Virtual Memory", "Paging, TLB, page replacement"),
    "5": ("filesystem", "File Systems", "Inodes, directories, file descriptors"),
    "6": ("syscalls", "System Calls", "User/kernel mode, syscall interface"),
}


def print_banner():
    """Print the main banner."""
    banner = f"""
{CYAN}╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   {BOLD}   ___  ____    _                          _             {RESET}{CYAN}   ║
║   {BOLD}  / _ \\/ ___|  | |    ___  __ _ _ __ _ __ (_)_ __   __ _ {RESET}{CYAN}   ║
║   {BOLD} | | | \\___ \\  | |   / _ \\/ _` | '__| '_ \\| | '_ \\ / _` |{RESET}{CYAN}   ║
║   {BOLD} | |_| |___) | | |__|  __/ (_| | |  | | | | | | | | (_| |{RESET}{CYAN}   ║
║   {BOLD}  \\___/|____/  |_____\\___|\\__,_|_|  |_| |_|_|_| |_|\\__, |{RESET}{CYAN}   ║
║   {BOLD}                                                   |___/ {RESET}{CYAN}   ║
║                                                                  ║
║   {YELLOW}Interactive Operating System Concepts Explorer{RESET}{CYAN}              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)


def print_menu():
    """Print the main menu."""
    print(f"\n{BOLD}  Available Modules:{RESET}\n")

    for key, (_, name, desc) in MODULES.items():
        print(f"    {CYAN}{key}.{RESET} {name}")
        print(f"       {DIM}{desc}{RESET}")
        print()

    print(f"    {CYAN}q.{RESET} Quit")
    print()


def run_module(module_name):
    """Run a specific module."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, f"{module_name}.py")

    if os.path.exists(script_path):
        subprocess.run([sys.executable, script_path])
    else:
        print(f"{RED}Module not found: {module_name}{RESET}")


def show_overview():
    """Show an overview of OS concepts."""
    print(f"""
{BOLD}{'═' * 65}{RESET}
{BOLD}           OPERATING SYSTEM OVERVIEW{RESET}
{BOLD}{'═' * 65}{RESET}

    {CYAN}What is an Operating System?{RESET}

    An OS is software that manages computer hardware and provides
    services for programs. It acts as an intermediary between users
    and hardware.

    {YELLOW}┌─────────────────────────────────────────────────────────────┐
    │                      User Programs                        │
    │                 (Applications, Shells)                    │
    ├─────────────────────────────────────────────────────────────┤
    │                    System Libraries                       │
    │                    (libc, libm, etc)                       │
    ├─────────────────────────────────────────────────────────────┤
    │                  {BOLD}Operating System{RESET}{YELLOW}                         │
    │    ┌──────────────────────────────────────────────────┐    │
    │    │                System Call Interface             │    │
    │    ├──────────────────────────────────────────────────┤    │
    │    │  Process    │  Memory    │  File    │  Device   │    │
    │    │  Manager    │  Manager   │  System  │  Drivers  │    │
    │    └──────────────────────────────────────────────────┘    │
    ├─────────────────────────────────────────────────────────────┤
    │                       Hardware                            │
    │             (CPU, Memory, Disk, Network)                  │
    └─────────────────────────────────────────────────────────────┘{RESET}

    {GREEN}Key OS Responsibilities:{RESET}

    1. {BOLD}Process Management{RESET}
       - Create, schedule, and terminate processes
       - Handle inter-process communication

    2. {BOLD}Memory Management{RESET}
       - Allocate and deallocate memory
       - Virtual memory and paging

    3. {BOLD}File System Management{RESET}
       - Organize and access files
       - Manage directories and permissions

    4. {BOLD}Device Management{RESET}
       - Control I/O devices
       - Provide uniform interface (drivers)

    5. {BOLD}Security & Protection{RESET}
       - User authentication
       - Resource access control
""")


def main():
    """Main entry point."""
    print_banner()

    while True:
        print_menu()
        choice = input(f"  {YELLOW}Select module (or 'o' for overview): {RESET}").strip().lower()

        if choice == "q":
            print(f"\n  {GREEN}Happy learning! 🎓{RESET}\n")
            break
        elif choice == "o":
            show_overview()
        elif choice in MODULES:
            module_name = MODULES[choice][0]
            run_module(module_name)
        else:
            print(f"  {RED}Invalid choice. Try again.{RESET}")


if __name__ == "__main__":
    main()
