#!/usr/bin/env python3
"""
Visualize CPU scheduling algorithms.
Interactive simulation of FCFS, SJF, Round Robin, and Priority scheduling.
"""

import sys
from dataclasses import dataclass
from typing import Optional
from collections import deque

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

COLORS = [RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN]


@dataclass
class Process:
    pid: int
    arrival: int
    burst: int
    priority: int = 0
    remaining: int = 0
    start_time: Optional[int] = None
    finish_time: Optional[int] = None

    def __post_init__(self):
        self.remaining = self.burst


def create_sample_processes():
    """Create sample process set."""
    return [
        Process(pid=1, arrival=0, burst=5, priority=2),
        Process(pid=2, arrival=1, burst=3, priority=1),
        Process(pid=3, arrival=2, burst=8, priority=3),
        Process(pid=4, arrival=3, burst=2, priority=4),
        Process(pid=5, arrival=4, burst=4, priority=2),
    ]


def draw_gantt_chart(schedule, total_time):
    """Draw a Gantt chart of the schedule."""
    print(f"\n{BOLD}Gantt Chart:{RESET}")

    # Top border
    chart = "  ┌"
    for _ in range(total_time):
        chart += "───┬"
    chart = chart[:-1] + "┐"
    print(chart)

    # Process bars
    row = "  │"
    for t in range(total_time):
        pid = schedule.get(t)
        if pid:
            color = COLORS[pid % len(COLORS)]
            row += f"{color} P{pid} {RESET}│"
        else:
            row += f"{DIM} -- {RESET}│"
    print(row)

    # Bottom border
    chart = "  └"
    for _ in range(total_time):
        chart += "───┴"
    chart = chart[:-1] + "┘"
    print(chart)

    # Time labels
    times = "   "
    for t in range(total_time + 1):
        times += f"{t:<4}"
    print(times)


def draw_ready_queue(queue, current_time, label="Ready Queue"):
    """Draw the ready queue state."""
    print(f"\n  {label} at t={current_time}: ", end="")
    if not queue:
        print(f"{DIM}[empty]{RESET}")
    else:
        print("[", end="")
        for i, p in enumerate(queue):
            color = COLORS[p.pid % len(COLORS)]
            print(f"{color}P{p.pid}(r={p.remaining}){RESET}", end="")
            if i < len(queue) - 1:
                print(", ", end="")
        print("]")


def fcfs_scheduler(processes):
    """First-Come-First-Served scheduling."""
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}    FIRST-COME-FIRST-SERVED (FCFS) SCHEDULING{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")

    procs = [Process(p.pid, p.arrival, p.burst, p.priority) for p in processes]
    procs.sort(key=lambda x: (x.arrival, x.pid))

    schedule = {}
    current_time = 0
    completed = []

    print(f"\n{DIM}Processes sorted by arrival time.{RESET}")
    print(f"{DIM}Non-preemptive: each process runs to completion.{RESET}\n")

    for p in procs:
        if current_time < p.arrival:
            current_time = p.arrival

        p.start_time = current_time

        print(f"  t={current_time}: {COLORS[p.pid % len(COLORS)]}P{p.pid}{RESET} starts (burst={p.burst})")

        for t in range(current_time, current_time + p.burst):
            schedule[t] = p.pid

        current_time += p.burst
        p.finish_time = current_time
        completed.append(p)

        print(f"  t={current_time}: {COLORS[p.pid % len(COLORS)]}P{p.pid}{RESET} completes")

    draw_gantt_chart(schedule, current_time)
    calculate_metrics(completed)


def sjf_scheduler(processes):
    """Shortest Job First (non-preemptive)."""
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}    SHORTEST JOB FIRST (SJF) SCHEDULING{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")

    procs = [Process(p.pid, p.arrival, p.burst, p.priority) for p in processes]
    schedule = {}
    current_time = 0
    completed = []
    ready_queue = []

    print(f"\n{DIM}Selects process with shortest burst time.{RESET}")
    print(f"{DIM}Non-preemptive: runs to completion once started.{RESET}\n")

    while len(completed) < len(procs):
        # Add newly arrived processes
        for p in procs:
            if p.arrival <= current_time and p not in ready_queue and p not in completed:
                ready_queue.append(p)

        if not ready_queue:
            current_time += 1
            continue

        # Select shortest job
        ready_queue.sort(key=lambda x: (x.burst, x.arrival))
        draw_ready_queue(ready_queue, current_time)

        p = ready_queue.pop(0)
        p.start_time = current_time

        print(f"  t={current_time}: {COLORS[p.pid % len(COLORS)]}P{p.pid}{RESET} selected (burst={p.burst})")

        for t in range(current_time, current_time + p.burst):
            schedule[t] = p.pid

        current_time += p.burst
        p.finish_time = current_time
        completed.append(p)

    draw_gantt_chart(schedule, current_time)
    calculate_metrics(completed)


def round_robin_scheduler(processes, quantum=2):
    """Round Robin scheduling."""
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}    ROUND ROBIN SCHEDULING (Quantum={quantum}){RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")

    procs = [Process(p.pid, p.arrival, p.burst, p.priority) for p in processes]
    schedule = {}
    current_time = 0
    completed = []
    ready_queue = deque()
    arrived = set()

    print(f"\n{DIM}Each process gets {quantum} time units, then preempted.{RESET}")
    print(f"{DIM}Preemptive: fair time sharing.{RESET}\n")

    while len(completed) < len(procs):
        # Add newly arrived processes
        for p in procs:
            if p.arrival <= current_time and p.pid not in arrived and p.remaining > 0:
                ready_queue.append(p)
                arrived.add(p.pid)

        if not ready_queue:
            current_time += 1
            continue

        draw_ready_queue(list(ready_queue), current_time)
        p = ready_queue.popleft()

        if p.start_time is None:
            p.start_time = current_time

        run_time = min(quantum, p.remaining)
        print(
            f"  t={current_time}: {COLORS[p.pid % len(COLORS)]}P{p.pid}{RESET} runs for {run_time} (remaining={p.remaining})"
        )

        for t in range(current_time, current_time + run_time):
            schedule[t] = p.pid

        current_time += run_time
        p.remaining -= run_time

        # Add processes that arrived during execution
        for proc in procs:
            if proc.arrival <= current_time and proc.pid not in arrived and proc.remaining > 0:
                ready_queue.append(proc)
                arrived.add(proc.pid)

        if p.remaining > 0:
            ready_queue.append(p)
            print(f"         {COLORS[p.pid % len(COLORS)]}P{p.pid}{RESET} preempted, back to queue")
        else:
            p.finish_time = current_time
            completed.append(p)
            print(f"         {COLORS[p.pid % len(COLORS)]}P{p.pid}{RESET} completed!")

    draw_gantt_chart(schedule, current_time)
    calculate_metrics(completed)


def priority_scheduler(processes):
    """Priority scheduling (non-preemptive, lower number = higher priority)."""
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}    PRIORITY SCHEDULING{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")

    procs = [Process(p.pid, p.arrival, p.burst, p.priority) for p in processes]
    schedule = {}
    current_time = 0
    completed = []
    ready_queue = []

    print(f"\n{DIM}Selects highest priority process (lower number = higher).{RESET}")
    print(f"{DIM}Non-preemptive.{RESET}\n")

    while len(completed) < len(procs):
        for p in procs:
            if p.arrival <= current_time and p not in ready_queue and p not in completed:
                ready_queue.append(p)

        if not ready_queue:
            current_time += 1
            continue

        # Select highest priority (lowest number)
        ready_queue.sort(key=lambda x: (x.priority, x.arrival))
        print(f"\n  Ready queue at t={current_time}:")
        for p in ready_queue:
            print(f"    {COLORS[p.pid % len(COLORS)]}P{p.pid}{RESET} priority={p.priority}")

        p = ready_queue.pop(0)
        p.start_time = current_time

        print(f"  → {COLORS[p.pid % len(COLORS)]}P{p.pid}{RESET} selected (priority={p.priority})")

        for t in range(current_time, current_time + p.burst):
            schedule[t] = p.pid

        current_time += p.burst
        p.finish_time = current_time
        completed.append(p)

    draw_gantt_chart(schedule, current_time)
    calculate_metrics(completed)


def calculate_metrics(completed):
    """Calculate and display scheduling metrics."""
    print(f"\n{BOLD}Performance Metrics:{RESET}")
    print(f"{'─' * 55}")
    print(f"  {'PID':<5} {'Arrival':<8} {'Burst':<7} {'Start':<7} {'Finish':<8} {'Wait':<6} {'TAT':<6}")
    print(f"{'─' * 55}")

    total_wait = 0
    total_tat = 0

    for p in sorted(completed, key=lambda x: x.pid):
        wait = p.start_time - p.arrival
        tat = p.finish_time - p.arrival
        total_wait += wait
        total_tat += tat

        color = COLORS[p.pid % len(COLORS)]
        print(f"  {color}P{p.pid}{RESET}    {p.arrival:<8} {p.burst:<7} {p.start_time:<7} {p.finish_time:<8} {wait:<6} {tat:<6}")

    print(f"{'─' * 55}")
    n = len(completed)
    print(f"  {CYAN}Average Waiting Time:    {total_wait/n:.2f}{RESET}")
    print(f"  {CYAN}Average Turnaround Time: {total_tat/n:.2f}{RESET}")


def show_process_table(processes):
    """Display the process table."""
    print(f"\n{BOLD}Process Table:{RESET}")
    print(f"{'─' * 40}")
    print(f"  {'PID':<5} {'Arrival':<10} {'Burst':<10} {'Priority':<10}")
    print(f"{'─' * 40}")
    for p in processes:
        color = COLORS[p.pid % len(COLORS)]
        print(f"  {color}P{p.pid}{RESET}    {p.arrival:<10} {p.burst:<10} {p.priority:<10}")
    print(f"{'─' * 40}")


def interactive_menu():
    """Interactive menu."""
    processes = create_sample_processes()

    while True:
        print(f"\n{BOLD}{'═' * 60}{RESET}")
        print(f"{BOLD}     CPU SCHEDULER SIMULATOR{RESET}")
        print(f"{BOLD}{'═' * 60}{RESET}")

        show_process_table(processes)

        print(f"""
  {CYAN}1.{RESET} First-Come-First-Served (FCFS)
  {CYAN}2.{RESET} Shortest Job First (SJF)
  {CYAN}3.{RESET} Round Robin (RR)
  {CYAN}4.{RESET} Priority Scheduling
  {CYAN}5.{RESET} Compare all algorithms
  {CYAN}6.{RESET} Custom processes
  {CYAN}q.{RESET} Quit
        """)

        choice = input(f"{YELLOW}Choose option: {RESET}").strip().lower()

        if choice == "1":
            fcfs_scheduler(processes)
        elif choice == "2":
            sjf_scheduler(processes)
        elif choice == "3":
            q = input("  Time quantum (default=2): ").strip()
            quantum = int(q) if q else 2
            round_robin_scheduler(processes, quantum)
        elif choice == "4":
            priority_scheduler(processes)
        elif choice == "5":
            fcfs_scheduler(processes)
            input(f"\n{DIM}Press Enter for next...{RESET}")
            sjf_scheduler(processes)
            input(f"\n{DIM}Press Enter for next...{RESET}")
            round_robin_scheduler(processes)
            input(f"\n{DIM}Press Enter for next...{RESET}")
            priority_scheduler(processes)
        elif choice == "6":
            processes = []
            n = int(input("  Number of processes: "))
            for i in range(n):
                print(f"  Process {i+1}:")
                arrival = int(input("    Arrival time: "))
                burst = int(input("    Burst time: "))
                priority = int(input("    Priority (1=highest): "))
                processes.append(Process(pid=i + 1, arrival=arrival, burst=burst, priority=priority))
        elif choice == "q":
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        processes = create_sample_processes()
        if sys.argv[1] == "--fcfs":
            fcfs_scheduler(processes)
        elif sys.argv[1] == "--sjf":
            sjf_scheduler(processes)
        elif sys.argv[1] == "--rr":
            round_robin_scheduler(processes)
        elif sys.argv[1] == "--priority":
            priority_scheduler(processes)
    else:
        interactive_menu()
