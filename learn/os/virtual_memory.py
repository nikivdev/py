#!/usr/bin/env python3
"""
Visualize virtual memory, paging, and page replacement algorithms.
"""

import sys
from collections import OrderedDict

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


def visualize_address_translation():
    """Show how virtual addresses are translated to physical addresses."""
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}        VIRTUAL TO PHYSICAL ADDRESS TRANSLATION{RESET}")
    print(f"{BOLD}{'═' * 65}{RESET}\n")

    # Example: 16-bit virtual address, 4KB pages
    page_size = 4096  # 4KB
    page_bits = 12  # 2^12 = 4096

    virtual_addr = 0x5A3C  # Example virtual address

    page_num = virtual_addr >> page_bits
    offset = virtual_addr & (page_size - 1)

    print(f"  {CYAN}Virtual Address: 0x{virtual_addr:04X} ({virtual_addr}){RESET}")
    print(f"  Page Size: {page_size} bytes (2^{page_bits})")
    print()

    # Binary breakdown
    binary = f"{virtual_addr:016b}"
    print(f"  Binary: {binary}")
    print(f"          {YELLOW}{'─' * 4}{RESET} {GREEN}{'─' * 12}{RESET}")
    print(f"          {YELLOW}Page{RESET}  {GREEN}Offset{RESET}")
    print()

    print(f"  {YELLOW}Page Number:{RESET} {page_num} (0x{page_num:X})")
    print(f"  {GREEN}Offset:{RESET}      {offset} (0x{offset:03X})")
    print()

    # Page table lookup
    page_table = {
        0: {"frame": 3, "valid": True},
        1: {"frame": 7, "valid": True},
        2: {"frame": None, "valid": False},  # Page fault!
        3: {"frame": 1, "valid": True},
        4: {"frame": 5, "valid": True},
        5: {"frame": 2, "valid": True},
    }

    print(f"{BOLD}  Page Table:{RESET}")
    print("  ┌─────────┬─────────┬─────────┐")
    print("  │  Page   │  Frame  │  Valid  │")
    print("  ├─────────┼─────────┼─────────┤")
    for pn, entry in page_table.items():
        marker = f" {CYAN}←{RESET}" if pn == page_num else ""
        valid_str = f"{GREEN}Yes{RESET}" if entry["valid"] else f"{RED}No{RESET}"
        frame_str = str(entry["frame"]) if entry["frame"] is not None else "-"
        print(f"  │    {pn}    │    {frame_str}    │   {valid_str}  │{marker}")
    print("  └─────────┴─────────┴─────────┘")
    print()

    if page_table.get(page_num, {}).get("valid"):
        frame_num = page_table[page_num]["frame"]
        physical_addr = (frame_num << page_bits) | offset

        print(f"  {GREEN}Page {page_num} → Frame {frame_num}{RESET}")
        print(f"  {GREEN}Physical Address: 0x{physical_addr:04X} ({physical_addr}){RESET}")
        print()
        print("  Physical = (Frame × PageSize) + Offset")
        print(f"           = ({frame_num} × {page_size}) + {offset}")
        print(f"           = {frame_num * page_size} + {offset}")
        print(f"           = {physical_addr}")
    else:
        print(f"  {RED}PAGE FAULT! Page {page_num} not in memory.{RESET}")


def visualize_page_table_structure():
    """Show page table entry structure."""
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}           PAGE TABLE ENTRY STRUCTURE{RESET}")
    print(f"{BOLD}{'═' * 65}{RESET}\n")

    print("  A typical Page Table Entry (PTE):")
    print()
    print("  ┌────────────────────────────────────────────────────────────┐")
    print(f"  │{CYAN}Frame Number{RESET}│{YELLOW}X{RESET}│{GREEN}R{RESET}│{MAGENTA}W{RESET}│{BLUE}U{RESET}│{RED}P{RESET}│{DIM}Reserved{RESET}│")
    print("  │  (20 bits) │1│1│1│1│1│ (7 bits)│")
    print("  └────────────────────────────────────────────────────────────┘")
    print()
    print(f"  {RED}P{RESET} - Present/Valid:     Is the page in physical memory?")
    print(f"  {MAGENTA}W{RESET} - Write:             Is writing allowed?")
    print(f"  {GREEN}R{RESET} - Referenced:        Has the page been accessed?")
    print(f"  {YELLOW}X{RESET} - Execute:           Can execute code from this page?")
    print(f"  {BLUE}U{RESET} - User:              Accessible in user mode?")
    print()

    # Example PTE
    print(f"  {BOLD}Example Entry:{RESET}")
    print("  ┌─────────────────────────────────────────────┐")
    print("  │ Frame: 0x1A3  │ X=1 │ R=1 │ W=0 │ U=1 │ P=1 │")
    print("  └─────────────────────────────────────────────┘")
    print("  → Page maps to frame 0x1A3")
    print("  → Executable, has been read, read-only, user-accessible, valid")


def visualize_tlb():
    """Visualize Translation Lookaside Buffer."""
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}        TLB (Translation Lookaside Buffer){RESET}")
    print(f"{BOLD}{'═' * 65}{RESET}\n")

    print(f"  {DIM}TLB is a cache for page table entries - much faster than RAM!{RESET}")
    print()

    # Memory hierarchy timing
    print(f"  {BOLD}Access Times:{RESET}")
    print("  ┌──────────────────────────────────────────┐")
    print(f"  │ TLB Hit:     ~1 cycle    {GREEN}████{RESET}             │")
    print(f"  │ Page Table: ~100 cycles {YELLOW}████████████████{RESET} │")
    print(f"  │ Page Fault: ~millions   {RED}[disk access]{RESET}     │")
    print("  └──────────────────────────────────────────┘")
    print()

    # TLB structure
    tlb_entries = [
        {"page": 5, "frame": 2, "valid": True},
        {"page": 1, "frame": 7, "valid": True},
        {"page": 3, "frame": 1, "valid": True},
        {"page": None, "frame": None, "valid": False},
    ]

    print(f"  {BOLD}TLB Contents (small, fully-associative cache):{RESET}")
    print("  ┌─────────┬─────────┬─────────┐")
    print("  │  Page   │  Frame  │  Valid  │")
    print("  ├─────────┼─────────┼─────────┤")
    for entry in tlb_entries:
        valid_str = f"{GREEN}Yes{RESET}" if entry["valid"] else f"{RED}No{RESET}"
        page_str = str(entry["page"]) if entry["page"] is not None else "-"
        frame_str = str(entry["frame"]) if entry["frame"] is not None else "-"
        print(f"  │    {page_str}    │    {frame_str}    │   {valid_str}  │")
    print("  └─────────┴─────────┴─────────┘")
    print()

    # Access flow
    print(f"  {BOLD}Address Translation Flow:{RESET}")
    print(f"""
           Virtual Address
                 │
                 ▼
         ┌───────────────┐
         │   Check TLB   │
         └───────┬───────┘
                 │
        ┌────────┴────────┐
        │                 │
     {GREEN}TLB Hit{RESET}           {YELLOW}TLB Miss{RESET}
        │                 │
        ▼                 ▼
   {GREEN}Use Frame{RESET}      ┌─────────────┐
        │         │ Page Table  │
        │         │   Lookup    │
        │         └──────┬──────┘
        │                │
        │         ┌──────┴──────┐
        │         │             │
        │      {GREEN}Found{RESET}        {RED}Not Found{RESET}
        │         │             │
        │    Update TLB    {RED}PAGE FAULT{RESET}
        │         │             │
        └────┬────┘        Load from
             │              Disk
             ▼
      Physical Address
    """)


def fifo_page_replacement(pages, frames):
    """FIFO page replacement algorithm."""
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}    FIFO PAGE REPLACEMENT (Frames={frames}){RESET}")
    print(f"{BOLD}{'═' * 65}{RESET}\n")

    memory = []
    page_faults = 0
    history = []

    for page in pages:
        fault = False
        evicted = None

        if page not in memory:
            fault = True
            page_faults += 1
            if len(memory) >= frames:
                evicted = memory.pop(0)
            memory.append(page)

        history.append({"page": page, "memory": memory.copy(), "fault": fault, "evicted": evicted})

    # Visualization
    _draw_page_replacement_table(pages, history, frames, "FIFO", page_faults)


def lru_page_replacement(pages, frames):
    """LRU page replacement algorithm."""
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}    LRU PAGE REPLACEMENT (Frames={frames}){RESET}")
    print(f"{BOLD}{'═' * 65}{RESET}\n")

    memory = OrderedDict()
    page_faults = 0
    history = []

    for page in pages:
        fault = False
        evicted = None

        if page in memory:
            memory.move_to_end(page)
        else:
            fault = True
            page_faults += 1
            if len(memory) >= frames:
                evicted, _ = memory.popitem(last=False)
            memory[page] = True

        history.append({"page": page, "memory": list(memory.keys()), "fault": fault, "evicted": evicted})

    _draw_page_replacement_table(pages, history, frames, "LRU", page_faults)


def optimal_page_replacement(pages, frames):
    """Optimal page replacement algorithm (Belady's)."""
    print(f"\n{BOLD}{'═' * 65}{RESET}")
    print(f"{BOLD}    OPTIMAL PAGE REPLACEMENT (Frames={frames}){RESET}")
    print(f"{BOLD}{'═' * 65}{RESET}\n")

    memory = []
    page_faults = 0
    history = []

    for i, page in enumerate(pages):
        fault = False
        evicted = None

        if page not in memory:
            fault = True
            page_faults += 1
            if len(memory) >= frames:
                # Find page used furthest in future
                future_use = {}
                for m in memory:
                    try:
                        future_use[m] = pages[i + 1 :].index(m)
                    except ValueError:
                        future_use[m] = float("inf")
                evicted = max(future_use, key=future_use.get)
                memory.remove(evicted)
            memory.append(page)

        history.append({"page": page, "memory": memory.copy(), "fault": fault, "evicted": evicted})

    _draw_page_replacement_table(pages, history, frames, "Optimal", page_faults)


def _draw_page_replacement_table(pages, history, frames, algo, faults):
    """Draw the page replacement visualization table."""
    print(f"  Reference String: {' '.join(str(p) for p in pages)}")
    print()

    # Header
    header = "  Time: "
    for i in range(len(pages)):
        header += f" {i:^3}"
    print(header)

    header = "  Page: "
    for p in pages:
        header += f" {p:^3}"
    print(header)
    print(f"  {'─' * (7 + len(pages) * 4)}")

    # Frame rows
    for f in range(frames):
        row = f"  F{f}:   "
        for h in history:
            if f < len(h["memory"]):
                page = h["memory"][f]
                if h["fault"] and page == h["page"]:
                    row += f" {GREEN}{page:^3}{RESET}"
                else:
                    row += f" {page:^3}"
            else:
                row += f" {DIM} - {RESET}"
        print(row)

    print(f"  {'─' * (7 + len(pages) * 4)}")

    # Fault row
    fault_row = "  Fault:"
    for h in history:
        if h["fault"]:
            fault_row += f" {RED} F {RESET}"
        else:
            fault_row += f" {GREEN} H {RESET}"
    print(fault_row)

    if any(h["evicted"] for h in history):
        evict_row = "  Evict:"
        for h in history:
            if h["evicted"]:
                evict_row += f" {YELLOW} {h['evicted']} {RESET}"
            else:
                evict_row += "    "
        print(evict_row)

    print()
    print(f"  {CYAN}Total Page Faults: {faults}{RESET}")
    print(f"  {CYAN}Hit Rate: {(len(pages) - faults) / len(pages) * 100:.1f}%{RESET}")


def interactive_menu():
    """Interactive menu."""
    while True:
        print(f"\n{BOLD}{'═' * 65}{RESET}")
        print(f"{BOLD}     VIRTUAL MEMORY EXPLORER{RESET}")
        print(f"{BOLD}{'═' * 65}{RESET}")
        print(f"""
  {CYAN}1.{RESET} Address translation demo
  {CYAN}2.{RESET} Page table entry structure
  {CYAN}3.{RESET} TLB visualization
  {CYAN}4.{RESET} FIFO page replacement
  {CYAN}5.{RESET} LRU page replacement
  {CYAN}6.{RESET} Optimal page replacement
  {CYAN}7.{RESET} Compare all algorithms
  {CYAN}q.{RESET} Quit
        """)

        choice = input(f"{YELLOW}Choose option: {RESET}").strip().lower()

        # Sample reference string
        sample_pages = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
        frames = 3

        if choice == "1":
            visualize_address_translation()
        elif choice == "2":
            visualize_page_table_structure()
        elif choice == "3":
            visualize_tlb()
        elif choice == "4":
            fifo_page_replacement(sample_pages, frames)
        elif choice == "5":
            lru_page_replacement(sample_pages, frames)
        elif choice == "6":
            optimal_page_replacement(sample_pages, frames)
        elif choice == "7":
            print(f"\n{DIM}Using reference string: {sample_pages}{RESET}")
            fifo_page_replacement(sample_pages, frames)
            input(f"\n{DIM}Press Enter for LRU...{RESET}")
            lru_page_replacement(sample_pages, frames)
            input(f"\n{DIM}Press Enter for Optimal...{RESET}")
            optimal_page_replacement(sample_pages, frames)
        elif choice == "q":
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        visualize_address_translation()
        visualize_tlb()
    else:
        interactive_menu()
