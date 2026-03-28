#!/usr/bin/env python3
"""
Visualize process creation with fork().
Shows parent/child relationship and address space copying.
"""

import os
import sys
import time
import multiprocessing as mp

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


def draw_process_box(title, pid, ppid, color, extra_info=None):
    """Draw a process visualization box."""
    width = 40
    print(f"{color}┌{'─' * width}┐{RESET}")
    print(f"{color}│{BOLD} {title:^{width-2}} {RESET}{color}│{RESET}")
    print(f"{color}├{'─' * width}┤{RESET}")
    print(f"{color}│{RESET}  PID:  {pid:<{width-10}}{color}│{RESET}")
    print(f"{color}│{RESET}  PPID: {ppid:<{width-10}}{color}│{RESET}")
    if extra_info:
        for key, val in extra_info.items():
            print(f"{color}│{RESET}  {key}: {str(val):<{width-len(key)-5}}{color}│{RESET}")
    print(f"{color}└{'─' * width}┘{RESET}")


def draw_memory_section(label, content, color):
    """Draw a memory section."""
    width = 30
    print(f"{color}  ┌{'─' * width}┐{RESET}")
    print(f"{color}  │ {label:<{width-2}}│{RESET}")
    print(f"{color}  │ {content:<{width-2}}│{RESET}")
    print(f"{color}  └{'─' * width}┘{RESET}")


def visualize_fork_concept():
    """Visualize the fork concept without actually forking."""
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}        FORK() - PROCESS CREATION CONCEPT{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")

    # Simulated parent
    print(f"{YELLOW}BEFORE fork():{RESET}\n")
    draw_process_box("Parent Process", os.getpid(), os.getppid(), BLUE)

    print(f"\n{DIM}  Memory (Address Space):{RESET}")
    draw_memory_section("TEXT", "Program code (read-only)", CYAN)
    draw_memory_section("DATA", "global_var = 100", MAGENTA)
    draw_memory_section("HEAP", "allocated_data = [...]", GREEN)
    draw_memory_section("STACK", "local_var = 42", RED)

    print(f"\n{YELLOW}══════════════ fork() called ══════════════{RESET}\n")

    print(f"{DIM}The kernel:{RESET}")
    print("  1. Creates new process entry (PCB)")
    print("  2. Copies parent's address space (Copy-on-Write)")
    print("  3. Child gets PID, parent gets child's PID")
    print()

    print(f"{YELLOW}AFTER fork():{RESET}\n")

    # Side by side visualization
    parent_lines = [
        f"{BLUE}┌────────────────────────────────────────┐{RESET}",
        f"{BLUE}│{BOLD}          Parent Process               {RESET}{BLUE}│{RESET}",
        f"{BLUE}├────────────────────────────────────────┤{RESET}",
        f"{BLUE}│{RESET}  PID:  {os.getpid():<30}{BLUE}│{RESET}",
        f"{BLUE}│{RESET}  fork() returns: child_pid           {BLUE}│{RESET}",
        f"{BLUE}├────────────────────────────────────────┤{RESET}",
        f"{BLUE}│{CYAN}  TEXT  {RESET}│{MAGENTA} DATA {RESET}│{GREEN} HEAP {RESET}│{RED} STACK {RESET}{BLUE}│{RESET}",
        f"{BLUE}└────────────────────────────────────────┘{RESET}",
    ]

    child_lines = [
        f"{GREEN}┌────────────────────────────────────────┐{RESET}",
        f"{GREEN}│{BOLD}           Child Process               {RESET}{GREEN}│{RESET}",
        f"{GREEN}├────────────────────────────────────────┤{RESET}",
        f"{GREEN}│{RESET}  PID:  (new pid)                     {GREEN}│{RESET}",
        f"{GREEN}│{RESET}  fork() returns: 0                   {GREEN}│{RESET}",
        f"{GREEN}├────────────────────────────────────────┤{RESET}",
        f"{GREEN}│{CYAN}  TEXT  {RESET}│{MAGENTA} DATA {RESET}│{GREEN} HEAP {RESET}│{RED} STACK {RESET}{GREEN}│{RESET}",
        f"{GREEN}└────────────────────────────────────────┘{RESET}",
    ]

    for p, c in zip(parent_lines, child_lines):
        print(f"  {p}    {c}")

    print(f"\n{DIM}Both processes continue from the same point,")
    print(f"but with different return values from fork().{RESET}")


def visualize_copy_on_write():
    """Visualize Copy-on-Write mechanism."""
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}        COPY-ON-WRITE (COW) MECHANISM{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")

    print(f"{YELLOW}Initial state after fork():{RESET}")
    print(f"{DIM}Both processes share the same physical pages (read-only){RESET}\n")

    # Shared pages visualization
    print(f"  {BLUE}Parent{RESET}                    {GREEN}Child{RESET}")
    print("  ┌─────────┐              ┌─────────┐")
    print("  │ Page 1  │──────┬───────│ Page 1  │")
    print("  │ Page 2  │──────┼───────│ Page 2  │")
    print("  │ Page 3  │──────┼───────│ Page 3  │")
    print("  └─────────┘      │       └─────────┘")
    print("                   ▼")
    print(f"            {CYAN}Physical Memory{RESET}")
    print("            ┌─────────────┐")
    print("            │   Page 1    │")
    print("            │   Page 2    │")
    print("            │   Page 3    │")
    print("            └─────────────┘")

    print(f"\n{YELLOW}After child writes to Page 2:{RESET}")
    print(f"{DIM}Only the modified page is copied{RESET}\n")

    print(f"  {BLUE}Parent{RESET}                    {GREEN}Child{RESET}")
    print("  ┌─────────┐              ┌─────────┐")
    print("  │ Page 1  │──────┬───────│ Page 1  │")
    print(f"  │ Page 2  │───┐  │   ┌───│ Page 2' │ {RED}(modified){RESET}")
    print("  │ Page 3  │───┼──┴───┼───│ Page 3  │")
    print("  └─────────┘   │      │   └─────────┘")
    print("                ▼      ▼")
    print(f"            {CYAN}Physical Memory{RESET}")
    print("            ┌─────────────┐")
    print("            │   Page 1    │")
    print("            │   Page 2    │ (original)")
    print(f"            │   Page 2'   │ {RED}(copy for child){RESET}")
    print("            │   Page 3    │")
    print("            └─────────────┘")


def demo_multiprocessing():
    """Demonstrate actual process creation with multiprocessing."""
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}        LIVE PROCESS CREATION DEMO{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")

    shared_before = 100

    def child_task(name, shared_val):
        pid = os.getpid()
        ppid = os.getppid()
        # Child modifies its copy
        local_val = shared_val + 50
        print(f"\n{GREEN}Child '{name}':{RESET}")
        print(f"  PID: {pid}, Parent PID: {ppid}")
        print(f"  Received value: {shared_val}")
        print(f"  Modified locally to: {local_val}")
        time.sleep(0.5)
        return local_val

    print(f"{BLUE}Parent Process (PID: {os.getpid()}):{RESET}")
    print(f"  shared_before = {shared_before}")
    print(f"\n{YELLOW}Creating child processes...{RESET}")

    # Create multiple children
    with mp.Pool(3) as pool:
        results = pool.starmap(
            child_task, [("Child-1", shared_before), ("Child-2", shared_before), ("Child-3", shared_before)]
        )

    print(f"\n{BLUE}Back in parent:{RESET}")
    print(f"  shared_before still = {shared_before} (unchanged!)")
    print(f"  Children returned: {results}")
    print(f"\n{DIM}Each child got a copy of the variable.{RESET}")
    print(f"{DIM}Changes in children don't affect parent.{RESET}")


def show_process_tree():
    """Show current process tree."""
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}        CURRENT PROCESS HIERARCHY{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")

    current_pid = os.getpid()
    parent_pid = os.getppid()

    print(f"{DIM}(Some ancestor){RESET}")
    print("    │")
    print(f"    ├── {YELLOW}Shell (PPID of Python){RESET}")
    print(f"    │       PID: {parent_pid}")
    print("    │")
    print(f"    └── {CYAN}This Python Script{RESET}")
    print(f"            PID: {current_pid}")
    print(f"            PPID: {parent_pid}")


def interactive_menu():
    """Interactive menu."""
    while True:
        print(f"\n{BOLD}{'═' * 60}{RESET}")
        print(f"{BOLD}     PROCESS FORK EXPLORER{RESET}")
        print(f"{BOLD}{'═' * 60}{RESET}")
        print(f"""
  {CYAN}1.{RESET} Fork concept visualization
  {CYAN}2.{RESET} Copy-on-Write explained
  {CYAN}3.{RESET} Live multiprocessing demo
  {CYAN}4.{RESET} Current process tree
  {CYAN}q.{RESET} Quit
        """)

        choice = input(f"{YELLOW}Choose option: {RESET}").strip().lower()

        if choice == "1":
            visualize_fork_concept()
        elif choice == "2":
            visualize_copy_on_write()
        elif choice == "3":
            demo_multiprocessing()
        elif choice == "4":
            show_process_tree()
        elif choice == "q":
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        visualize_fork_concept()
        visualize_copy_on_write()
    else:
        interactive_menu()
