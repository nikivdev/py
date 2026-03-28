#!/usr/bin/env python3
"""
Explore system calls - the interface between user space and kernel.
"""

import os
import sys
import resource

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


def visualize_syscall_flow():
    """Show how system calls work."""
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}           SYSTEM CALL MECHANISM{RESET}")
    print(f"{BOLD}{'═' * 65}{RESET}\n")

    print(f"""
    {BLUE}User Space{RESET}                              {RED}Kernel Space{RESET}
    ┌─────────────────────────────────────┬─────────────────────────────┐
    │                                     │                             │
    │  {GREEN}Your Program{RESET}                        │                             │
    │       │                             │                             │
    │       │ calls printf("hello")       │                             │
    │       ▼                             │                             │
    │  {YELLOW}C Library (libc){RESET}                   │                             │
    │       │                             │                             │
    │       │ calls write(1, "hello", 5)  │                             │
    │       ▼                             │                             │
    │  {MAGENTA}System Call Wrapper{RESET}                │                             │
    │       │                             │                             │
    │       │ syscall(SYS_write, ...)     │                             │
    │       │                             │                             │
    │       │ ──── {CYAN}TRAP / INT 0x80{RESET} ────────► {RED}Kernel Entry Point{RESET}        │
    │       │      {DIM}(mode switch){RESET}            │       │                   │
    │       │                             │       ▼                   │
    │       │                             │  {RED}System Call Handler{RESET}      │
    │       │                             │       │                   │
    │       │                             │       │ sys_write()       │
    │       │                             │       │                   │
    │       │                             │       ▼                   │
    │       │                             │  {RED}Device Driver{RESET}            │
    │       │                             │       │                   │
    │       │ ◄─── {CYAN}Return{RESET} ─────────────────│ ◄─────┘                   │
    │       │                             │                             │
    │       ▼                             │                             │
    │   (continue)                        │                             │
    │                                     │                             │
    └─────────────────────────────────────┴─────────────────────────────┘
    """)

    print(f"  {BOLD}Key Points:{RESET}")
    print("  • User programs cannot directly access hardware/kernel memory")
    print("  • System calls are the ONLY way to request kernel services")
    print("  • Mode switch (user→kernel) is expensive (~1000s of cycles)")
    print("  • That's why buffering I/O is important!")


def show_common_syscalls():
    """Display common system calls by category."""
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}           COMMON SYSTEM CALLS{RESET}")
    print(f"{BOLD}{'═' * 65}{RESET}\n")

    syscalls = {
        "Process Control": [
            ("fork()", "Create new process"),
            ("exec()", "Execute a program"),
            ("exit()", "Terminate process"),
            ("wait()", "Wait for child process"),
            ("getpid()", "Get process ID"),
            ("kill()", "Send signal to process"),
        ],
        "File Operations": [
            ("open()", "Open a file"),
            ("close()", "Close file descriptor"),
            ("read()", "Read from file"),
            ("write()", "Write to file"),
            ("lseek()", "Move file pointer"),
            ("stat()", "Get file status"),
        ],
        "Directory Operations": [
            ("mkdir()", "Create directory"),
            ("rmdir()", "Remove directory"),
            ("chdir()", "Change directory"),
            ("getcwd()", "Get current directory"),
            ("readdir()", "Read directory entries"),
        ],
        "Memory Management": [
            ("brk()", "Change data segment size"),
            ("mmap()", "Map files/memory"),
            ("munmap()", "Unmap memory"),
            ("mprotect()", "Set memory protection"),
        ],
        "Network": [
            ("socket()", "Create socket"),
            ("bind()", "Bind to address"),
            ("listen()", "Listen for connections"),
            ("accept()", "Accept connection"),
            ("connect()", "Connect to server"),
            ("send()/recv()", "Send/receive data"),
        ],
    }

    for category, calls in syscalls.items():
        color = {"Process Control": GREEN, "File Operations": CYAN, "Directory Operations": YELLOW, "Memory Management": MAGENTA, "Network": BLUE}.get(
            category, RESET
        )
        print(f"  {color}{BOLD}{category}:{RESET}")
        for name, desc in calls:
            print(f"    {name:<15} {DIM}{desc}{RESET}")
        print()


def demo_syscalls():
    """Demonstrate actual system calls in Python."""
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}           LIVE SYSTEM CALL DEMOS{RESET}")
    print(f"{BOLD}{'═' * 65}{RESET}\n")

    # getpid
    print(f"  {CYAN}os.getpid(){RESET} → getpid() syscall")
    pid = os.getpid()
    print(f"    Result: {pid}")
    print()

    # getcwd
    print(f"  {CYAN}os.getcwd(){RESET} → getcwd() syscall")
    cwd = os.getcwd()
    print(f"    Result: {cwd}")
    print()

    # stat
    print(f"  {CYAN}os.stat(__file__){RESET} → stat() syscall")
    st = os.stat(__file__)
    print(f"    inode: {st.st_ino}, size: {st.st_size}, mode: {oct(st.st_mode)}")
    print()

    # getuid/getgid
    print(f"  {CYAN}os.getuid(), os.getgid(){RESET} → getuid(), getgid() syscalls")
    print(f"    UID: {os.getuid()}, GID: {os.getgid()}")
    print()

    # uname
    print(f"  {CYAN}os.uname(){RESET} → uname() syscall")
    uname = os.uname()
    print(f"    System: {uname.sysname} {uname.release}")
    print(f"    Machine: {uname.machine}")
    print()

    # resource limits (getrlimit)
    print(f"  {CYAN}resource.getrlimit(){RESET} → getrlimit() syscall")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"    Max open files: soft={soft}, hard={hard}")


def show_kernel_interaction():
    """Show how to see kernel info."""
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}           KERNEL INFORMATION{RESET}")
    print(f"{BOLD}{'═' * 65}{RESET}\n")

    uname = os.uname()

    print(f"  {BOLD}System Information (from uname syscall):{RESET}")
    print(f"  ┌{'─' * 50}┐")
    print(f"  │ {CYAN}OS Name:{RESET}     {uname.sysname:<36}│")
    print(f"  │ {CYAN}Hostname:{RESET}    {uname.nodename:<36}│")
    print(f"  │ {CYAN}Release:{RESET}     {uname.release:<36}│")
    print(f"  │ {CYAN}Version:{RESET}     {uname.version[:36]:<36}│")
    print(f"  │ {CYAN}Machine:{RESET}     {uname.machine:<36}│")
    print(f"  └{'─' * 50}┘")
    print()

    print(f"  {BOLD}Resource Limits (getrlimit syscall):{RESET}")
    limits = [
        (resource.RLIMIT_CPU, "CPU time (seconds)"),
        (resource.RLIMIT_FSIZE, "File size (bytes)"),
        (resource.RLIMIT_DATA, "Data segment (bytes)"),
        (resource.RLIMIT_STACK, "Stack size (bytes)"),
        (resource.RLIMIT_NOFILE, "Open files"),
        (resource.RLIMIT_NPROC, "Max processes"),
    ]

    print(f"  ┌{'─' * 50}┐")
    print(f"  │ {'Resource':<20} {'Soft':<12} {'Hard':<12} │")
    print(f"  ├{'─' * 50}┤")
    for limit, name in limits:
        try:
            soft, hard = resource.getrlimit(limit)
            soft_str = str(soft) if soft != resource.RLIM_INFINITY else "unlimited"
            hard_str = str(hard) if hard != resource.RLIM_INFINITY else "unlimited"
            print(f"  │ {name:<20} {soft_str:<12} {hard_str:<12} │")
        except (ValueError, OSError):
            pass
    print(f"  └{'─' * 50}┘")


def visualize_protection_rings():
    """Show CPU protection rings."""
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}           CPU PROTECTION RINGS{RESET}")
    print(f"{BOLD}{'═' * 65}{RESET}\n")

    print(f"""
                    Protection Rings (x86)

                     ┌──────────────────┐
                     │   {RED}Ring 0{RESET}          │
                     │   {RED}Kernel Mode{RESET}     │
                     │                  │
                     │  - Full hardware │
                     │    access        │
                     │  - All memory    │
                     │  - All I/O       │
               ┌─────┴──────────────────┴─────┐
               │       {YELLOW}Ring 1, 2{RESET}              │
               │       {DIM}(Rarely used){RESET}           │
               │       Device drivers         │
          ┌────┴──────────────────────────────┴────┐
          │            {GREEN}Ring 3{RESET}                     │
          │            {GREEN}User Mode{RESET}                  │
          │                                        │
          │   - Limited hardware access            │
          │   - Restricted memory (own space)      │
          │   - Must use syscalls for I/O          │
          │                                        │
          │   {CYAN}Your programs run here!{RESET}             │
          └────────────────────────────────────────┘

    {BOLD}Mode Transitions:{RESET}

    User Mode (Ring 3)              Kernel Mode (Ring 0)
         │                                  │
         │ ──── System Call ───────────►    │
         │ ──── Interrupt ─────────────►    │
         │ ──── Exception ─────────────►    │
         │                                  │
         │ ◄─── Return from syscall ────    │
         │ ◄─── Return from handler ────    │
    """)


def interactive_menu():
    """Interactive menu."""
    while True:
        print(f"\n{BOLD}{'═' * 65}{RESET}")
        print(f"{BOLD}     SYSTEM CALLS EXPLORER{RESET}")
        print(f"{BOLD}{'═' * 65}{RESET}")
        print(f"""
  {CYAN}1.{RESET} System call mechanism
  {CYAN}2.{RESET} Common system calls list
  {CYAN}3.{RESET} Live syscall demos
  {CYAN}4.{RESET} Kernel information
  {CYAN}5.{RESET} Protection rings
  {CYAN}q.{RESET} Quit
        """)

        choice = input(f"{YELLOW}Choose option: {RESET}").strip().lower()

        if choice == "1":
            visualize_syscall_flow()
        elif choice == "2":
            show_common_syscalls()
        elif choice == "3":
            demo_syscalls()
        elif choice == "4":
            show_kernel_interaction()
        elif choice == "5":
            visualize_protection_rings()
        elif choice == "q":
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        visualize_syscall_flow()
        show_common_syscalls()
    else:
        interactive_menu()
