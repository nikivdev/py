#!/usr/bin/env python3
"""
Visualize process memory layout: text, data, heap, stack sections.
Shows how memory is organized in a running process.
"""

import os
import sys

# ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"


def get_memory_info():
    """Get actual memory addresses of different sections."""
    # Code/Text section - address of a function
    code_addr = id(get_memory_info)

    # Data section - global variable
    global_var = "I'm in data section"
    data_addr = id(global_var)

    # Heap - dynamically allocated
    heap_list = [1, 2, 3, 4, 5]
    heap_addr = id(heap_list)

    # Stack - local variable
    stack_var = 42
    stack_addr = id(stack_var)

    return {
        "code": code_addr,
        "data": data_addr,
        "heap": heap_addr,
        "stack": stack_addr,
    }


def draw_memory_box(label, addr, color, content_lines, width=50):
    """Draw a memory section box."""
    print(f"{color}┌{'─' * width}┐{RESET}")
    print(f"{color}│{BOLD} {label:^{width-2}} {RESET}{color}│{RESET}")
    print(f"{color}│{DIM} 0x{addr:016x} {RESET}{color}{' ' * (width-22)}│{RESET}")
    print(f"{color}├{'─' * width}┤{RESET}")
    for line in content_lines:
        print(f"{color}│{RESET} {line:<{width-2}} {color}│{RESET}")
    print(f"{color}└{'─' * width}┘{RESET}")


def visualize_memory_layout():
    """Main visualization of process memory layout."""
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}       PROCESS MEMORY LAYOUT (PID: {os.getpid()}){RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")

    print(f"{DIM}High Address (0xFFFF...){RESET}")
    print(f"{DIM}    │{RESET}")
    print()

    # Stack (grows downward)
    draw_memory_box(
        "STACK",
        id(visualize_memory_layout.__code__),
        RED,
        [
            "Local variables",
            "Function parameters",
            "Return addresses",
            f"{DIM}↓ Grows downward{RESET}",
        ],
    )
    print(f"{DIM}    │{RESET}")
    print(f"{DIM}    ▼ (free space){RESET}")
    print(f"{DIM}    ▲ (free space){RESET}")
    print(f"{DIM}    │{RESET}")

    # Heap (grows upward)
    heap_obj = {"key": "value", "list": [1, 2, 3]}
    draw_memory_box(
        "HEAP",
        id(heap_obj),
        GREEN,
        [
            "Dynamic allocations (malloc/new)",
            "Objects, arrays, dictionaries",
            f"Example: {heap_obj}",
            f"{DIM}↑ Grows upward{RESET}",
        ],
    )
    print(f"{DIM}    │{RESET}")

    # BSS (uninitialized data)
    draw_memory_box(
        "BSS (Uninitialized Data)",
        0x0000600000000000,
        YELLOW,
        [
            "Global variables = 0",
            "Static variables = 0",
        ],
    )
    print(f"{DIM}    │{RESET}")

    # Data section
    global_str = "Hello, OS!"
    draw_memory_box(
        "DATA (Initialized)",
        id(global_str),
        MAGENTA,
        [
            "Global initialized variables",
            "Static initialized variables",
            f'Example: global_str = "{global_str}"',
        ],
    )
    print(f"{DIM}    │{RESET}")

    # Text/Code section
    draw_memory_box(
        "TEXT (Code)",
        id(visualize_memory_layout),
        CYAN,
        [
            "Executable instructions",
            "Read-only",
            "Shared between processes",
        ],
    )

    print()
    print(f"{DIM}    │{RESET}")
    print(f"{DIM}Low Address (0x0000...){RESET}")
    print()


def show_stack_frames():
    """Demonstrate stack frames with function calls."""
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}           STACK FRAMES VISUALIZATION{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")

    def level_3(x):
        frame = sys._getframe()
        print(f"{RED}┌─ Frame 3: level_3(){RESET}")
        print(f"{RED}│  Local: x = {x}{RESET}")
        print(f"{RED}│  Return to: {frame.f_back.f_code.co_name}(){RESET}")
        print(f"{RED}└─────────────────────{RESET}")
        return x * 2

    def level_2(x):
        frame = sys._getframe()
        print(f"{YELLOW}┌─ Frame 2: level_2(){RESET}")
        print(f"{YELLOW}│  Local: x = {x}{RESET}")
        print(f"{YELLOW}│  Return to: {frame.f_back.f_code.co_name}(){RESET}")
        print(f"{YELLOW}└─────────────────────{RESET}")
        return level_3(x + 10)

    def level_1(x):
        frame = sys._getframe()
        print(f"{GREEN}┌─ Frame 1: level_1(){RESET}")
        print(f"{GREEN}│  Local: x = {x}{RESET}")
        print(f"{GREEN}│  Return to: {frame.f_back.f_code.co_name}(){RESET}")
        print(f"{GREEN}└─────────────────────{RESET}")
        return level_2(x + 5)

    print(f"{DIM}Stack grows downward as functions are called:{RESET}\n")
    result = level_1(1)
    print(f"\n{CYAN}Result returned through stack: {result}{RESET}")


def show_heap_allocation():
    """Demonstrate heap allocations."""
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}           HEAP ALLOCATIONS{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")

    allocations = []

    print(f"{DIM}Allocating objects on the heap:{RESET}\n")

    for i in range(5):
        obj = [0] * (100 * (i + 1))  # Varying sizes
        addr = id(obj)
        size = sys.getsizeof(obj)
        allocations.append((addr, size, f"list_{i}"))
        print(f"{GREEN}  Allocated list_{i}: {RESET}")
        print(f"    Address: 0x{addr:016x}")
        print(f"    Size: {size} bytes")
        print()

    print(f"{YELLOW}Heap allocation pattern:{RESET}")
    for addr, size, name in sorted(allocations):
        bar_len = size // 100
        print(f"  0x{addr:012x} │{'█' * bar_len} {name} ({size}B)")


def interactive_menu():
    """Interactive menu for exploring memory concepts."""
    while True:
        print(f"\n{BOLD}{'═' * 60}{RESET}")
        print(f"{BOLD}     PROCESS MEMORY EXPLORER{RESET}")
        print(f"{BOLD}{'═' * 60}{RESET}")
        print(f"""
  {CYAN}1.{RESET} View memory layout
  {CYAN}2.{RESET} Stack frames demo
  {CYAN}3.{RESET} Heap allocations demo
  {CYAN}4.{RESET} Current process info
  {CYAN}q.{RESET} Quit
        """)

        choice = input(f"{YELLOW}Choose option: {RESET}").strip().lower()

        if choice == "1":
            visualize_memory_layout()
        elif choice == "2":
            show_stack_frames()
        elif choice == "3":
            show_heap_allocation()
        elif choice == "4":
            print(f"\n{CYAN}Process ID: {os.getpid()}{RESET}")
            print(f"{CYAN}Parent PID: {os.getppid()}{RESET}")
            print(f"{CYAN}Python executable: {sys.executable}{RESET}")
        elif choice == "q":
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        visualize_memory_layout()
        show_stack_frames()
    else:
        interactive_menu()
