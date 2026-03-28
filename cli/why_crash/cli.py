"""Collect diagnostics to explain a sudden Safe Mode boot or shell breakage."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

MAX_CRASH_CONTENT = 200_000


async def analyze_crash_file(path: Path) -> int:
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

    content = path.read_text(errors="ignore")
    if len(content) > MAX_CRASH_CONTENT:
        content = content[:MAX_CRASH_CONTENT] + "\n... (truncated)"

    prompt = (
        "You are a crash analysis expert. Analyze the following crash report and explain:\n"
        "1. What process crashed and why\n"
        "2. The root cause (interpret the exception type, faulting address, stack trace)\n"
        "3. Which library/component is responsible\n"
        "4. Actionable steps to fix or avoid the crash\n\n"
        "Be concise and direct. Here is the crash report:\n\n"
        f"```\n{content}\n```"
    )

    options = ClaudeAgentOptions(
        system_prompt="You are a systems debugging expert. Analyze crash reports concisely.",
        max_turns=1,
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    sys.stdout.write(block.text)
    sys.stdout.write("\n")
    return 0


@dataclass(frozen=True)
class CommandResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class PanicReport:
    path: Path
    mtime: float
    panic_string: str | None
    bsd_process: str | None
    os_version: str | None
    kernel_version: str | None
    kext_backtrace: list[str]
    last_loaded_kext: str | None
    last_unloaded_kext: str | None


@dataclass(frozen=True)
class HardenAction:
    path: Path
    status: str
    detail: str


def run_cmd(cmd: list[str], *, timeout: int = 20) -> CommandResult:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return CommandResult(cmd, 127, "", str(exc), False)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return CommandResult(cmd, 124, stdout, stderr, True)

    return CommandResult(cmd, result.returncode, result.stdout, result.stderr, False)


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def write_text(path: Path, content: str, *, mode: int = 0o755) -> None:
    path.write_text(content)
    path.chmod(mode)


def extract_first(lines: Iterable[str]) -> str | None:
    for line in lines:
        text = line.strip()
        if text:
            return text
    return None


def parse_panic_report(path: Path, *, mtime: float | None = None) -> PanicReport:
    text = read_text(path)
    panic_string = None
    bsd_process = None
    os_version = None
    kernel_version = None
    kext_backtrace: list[str] = []
    last_loaded_kext = None
    last_unloaded_kext = None

    panic_match = re.search(r"panic\([^\)]*\):\s*(.+)", text)
    if panic_match:
        panic_string = panic_match.group(1).strip()
    else:
        panic_match = re.search(r"panicString:\s*(.+)", text)
        if panic_match:
            panic_string = panic_match.group(1).strip()

    bsd_match = re.search(
        r"BSD process name corresponding to current thread:\s*(.+)", text
    )
    if bsd_match:
        bsd_process = bsd_match.group(1).strip()

    os_match = re.search(r"(?:Mac OS version|OS version):\s*(.+)", text)
    if os_match:
        os_version = os_match.group(1).strip()

    kernel_match = re.search(r"Kernel version:\s*(.+)", text)
    if kernel_match:
        kernel_version = kernel_match.group(1).strip()

    last_loaded = re.search(r"last loaded kext at .*?:\s*(.+)", text)
    if last_loaded:
        last_loaded_kext = last_loaded.group(1).strip()

    last_unloaded = re.search(r"last unloaded kext at .*?:\s*(.+)", text)
    if last_unloaded:
        last_unloaded_kext = last_unloaded.group(1).strip()

    if "Kernel Extensions in backtrace:" in text:
        backtrace_lines: list[str] = []
        capture = False
        for line in text.splitlines():
            if line.strip().startswith("Kernel Extensions in backtrace:"):
                capture = True
                continue
            if capture:
                if not line.strip():
                    break
                backtrace_lines.append(line.strip())
        kext_backtrace = backtrace_lines

    report_mtime = mtime
    if report_mtime is None:
        try:
            report_mtime = path.stat().st_mtime
        except OSError:
            report_mtime = 0.0

    return PanicReport(
        path=path,
        mtime=report_mtime,
        panic_string=panic_string,
        bsd_process=bsd_process,
        os_version=os_version,
        kernel_version=kernel_version,
        kext_backtrace=kext_backtrace,
        last_loaded_kext=last_loaded_kext,
        last_unloaded_kext=last_unloaded_kext,
    )


def collect_panic_reports(max_reports: int) -> list[PanicReport]:
    diag_dir = Path("/Library/Logs/DiagnosticReports")
    if not diag_dir.exists():
        return []

    reports: list[tuple[Path, float]] = []
    for path in diag_dir.glob("*.panic"):
        try:
            reports.append((path, path.stat().st_mtime))
        except OSError:
            continue

    reports.sort(key=lambda item: item[1], reverse=True)
    return [parse_panic_report(path, mtime=mtime) for path, mtime in reports[:max_reports]]


def format_command_result(result: CommandResult, *, max_lines: int) -> list[str]:
    lines: list[str] = []
    header = " ".join(result.cmd)
    if result.timed_out:
        lines.append(f"[timeout] {header}")
    elif result.returncode != 0:
        lines.append(f"[exit {result.returncode}] {header}")
    else:
        lines.append(header)

    stdout_lines = [line.rstrip() for line in result.stdout.splitlines() if line.strip()]
    stderr_lines = [line.rstrip() for line in result.stderr.splitlines() if line.strip()]

    if stdout_lines:
        lines.append("stdout:")
        lines.extend(stdout_lines[:max_lines])
        if len(stdout_lines) > max_lines:
            lines.append(f"... ({len(stdout_lines) - max_lines} more lines)")
    if stderr_lines:
        lines.append("stderr:")
        lines.extend(stderr_lines[:max_lines])
        if len(stderr_lines) > max_lines:
            lines.append(f"... ({len(stderr_lines) - max_lines} more lines)")
    return lines


def render_report(
    *,
    hours: int,
    max_lines: int,
    include_disk: bool,
    include_proxy: bool,
    include_launch: bool,
    include_logs: bool,
    panic_reports: list[PanicReport],
    log_results: list[CommandResult],
    extra_results: list[CommandResult],
) -> str:
    now = dt.datetime.now().astimezone()
    lines: list[str] = []
    lines.append("why-crash report")
    lines.append(f"generated: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append(f"host: {platform.node()}")
    lines.append(f"os: {platform.platform()}")
    lines.append("")

    lines.append("summary")
    if panic_reports:
        latest = panic_reports[0]
        lines.append(f"- panic reports found: {len(panic_reports)} (latest: {latest.path.name})")
        if latest.panic_string:
            lines.append(f"- latest panic: {latest.panic_string}")
        if latest.bsd_process:
            lines.append(f"- latest bsd process: {latest.bsd_process}")
    else:
        lines.append("- panic reports found: 0")

    lines.append(f"- log window: last {hours}h")
    lines.append("")

    lines.append("panic reports")
    if not panic_reports:
        lines.append("(none)")
    else:
        for report in panic_reports:
            ts = dt.datetime.fromtimestamp(report.mtime).astimezone()
            lines.append(f"- {report.path.name} ({ts.strftime('%Y-%m-%d %H:%M:%S %Z')})")
            if report.panic_string:
                lines.append(f"  panic: {report.panic_string}")
            if report.bsd_process:
                lines.append(f"  bsd process: {report.bsd_process}")
            if report.os_version:
                lines.append(f"  os: {report.os_version}")
            if report.kernel_version:
                lines.append(f"  kernel: {report.kernel_version}")
            if report.last_loaded_kext:
                lines.append(f"  last loaded kext: {report.last_loaded_kext}")
            if report.last_unloaded_kext:
                lines.append(f"  last unloaded kext: {report.last_unloaded_kext}")
            if report.kext_backtrace:
                lines.append("  kext backtrace:")
                lines.extend(f"    {entry}" for entry in report.kext_backtrace)

    lines.append("")
    if include_logs:
        lines.append("log show (safe mode / panic / shutdown cause)")
        if not log_results:
            lines.append("(skipped)")
        else:
            for result in log_results:
                lines.extend(format_command_result(result, max_lines=max_lines))
                lines.append("")

    if include_disk:
        lines.append("diskutil verifyVolume /")
        for result in extra_results:
            if result.cmd[:2] == ["diskutil", "verifyVolume"]:
                lines.extend(format_command_result(result, max_lines=max_lines))
                lines.append("")

    if include_proxy:
        lines.append("proxy settings")
        for result in extra_results:
            if result.cmd[:2] in (["scutil", "--proxy"], ["networksetup", "-getwebproxy"], ["networksetup", "-getsecurewebproxy"], ["networksetup", "-getautoproxyurl"], ["networksetup", "-getproxybypassdomains"]):
                lines.extend(format_command_result(result, max_lines=max_lines))
                lines.append("")

    if include_launch:
        lines.append("launch agents/daemons")
        for result in extra_results:
            if result.cmd and result.cmd[0] == "ls" and "Launch" in " ".join(result.cmd):
                lines.extend(format_command_result(result, max_lines=max_lines))
                lines.append("")

    lines.append("shell info")
    for result in extra_results:
        if result.cmd[:2] in (
            ["dscl", "."],
            ["which", "fish"],
            ["fish", "--version"],
            ["printenv", "SHELL"],
            ["defaults", "read"],
        ):
            lines.extend(format_command_result(result, max_lines=max_lines))
            lines.append("")

    lines.append("recent reboots")
    for result in extra_results:
        if result.cmd and result.cmd[0] == "last":
            lines.extend(format_command_result(result, max_lines=max_lines))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="why-crash",
        description="Analyze crash reports or collect diagnostics for Safe Mode boots.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        default=None,
        help="Path to a crash report file to analyze with Claude.",
    )
    parser.add_argument(
        "--harden",
        action="store_true",
        help="Write safe wrappers (fish/proxy-off) into ~/bin for resilience.",
    )
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=Path.home() / "bin",
        help="Where to install wrappers when --harden is set (default: ~/bin).",
    )
    parser.add_argument(
        "--stable-fish",
        type=Path,
        default=Path("/opt/homebrew/bin/fish"),
        help="Preferred fish binary for the wrapper (default: /opt/homebrew/bin/fish).",
    )
    parser.add_argument(
        "--dev-fish",
        type=Path,
        default=Path.home() / ".local/bin/fish",
        help="Dev fish binary for opt-in use (default: ~/.local/bin/fish).",
    )
    parser.add_argument(
        "--no-fish-wrapper",
        action="store_true",
        help="Skip writing the fish safety wrapper when --harden is set.",
    )
    parser.add_argument(
        "--no-proxy-script",
        action="store_true",
        help="Skip writing the proxy-off helper when --harden is set.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing wrapper files when --harden is set.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="How far back to query logs (default: 24).",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=200,
        help="Max lines per command section (default: 200).",
    )
    parser.add_argument(
        "--max-panics",
        type=int,
        default=5,
        help="How many panic reports to include (default: 5).",
    )
    parser.add_argument(
        "--no-disk",
        action="store_true",
        help="Skip diskutil verifyVolume /.",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Skip proxy configuration checks.",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Skip LaunchAgents/LaunchDaemons listing.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Skip log show queries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write report to a file instead of stdout.",
    )
    return parser


def collect_proxy_settings() -> list[CommandResult]:
    results: list[CommandResult] = []
    results.append(run_cmd(["scutil", "--proxy"], timeout=10))

    services_result = run_cmd(["networksetup", "-listallnetworkservices"], timeout=10)
    results.append(services_result)
    services: list[str] = []
    if services_result.returncode == 0:
        for line in services_result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("An asterisk"):
                continue
            services.append(line.lstrip("* "))

    for service in services:
        results.append(run_cmd(["networksetup", "-getwebproxy", service], timeout=10))
        results.append(run_cmd(["networksetup", "-getsecurewebproxy", service], timeout=10))
        results.append(run_cmd(["networksetup", "-getautoproxyurl", service], timeout=10))
        results.append(run_cmd(["networksetup", "-getproxybypassdomains", service], timeout=10))
    return results


def collect_shell_info() -> list[CommandResult]:
    user = os.environ.get("USER", "")
    results = [
        run_cmd(["printenv", "SHELL"], timeout=5),
    ]
    if user:
        results.append(run_cmd(["dscl", ".", "-read", f"/Users/{user}", "UserShell"], timeout=5))
    results.append(run_cmd(["which", "fish"], timeout=5))
    results.append(run_cmd(["fish", "--version"], timeout=5))
    results.append(run_cmd(["defaults", "read", "-g", "AppleInterfaceStyle"], timeout=5))
    return results


def collect_launch_items() -> list[CommandResult]:
    return [
        run_cmd(["ls", "-la", str(Path.home() / "Library/LaunchAgents")], timeout=10),
        run_cmd(["ls", "-la", "/Library/LaunchAgents"], timeout=10),
        run_cmd(["ls", "-la", "/Library/LaunchDaemons"], timeout=10),
    ]


def collect_recent_reboots() -> list[CommandResult]:
    return [
        run_cmd(["last", "-20", "reboot"], timeout=10),
        run_cmd(["last", "-20", "shutdown"], timeout=10),
    ]


def collect_logs(hours: int) -> list[CommandResult]:
    predicate = (
        'eventMessage CONTAINS[c] "panic" '
        'OR eventMessage CONTAINS[c] "Safe Mode" '
        'OR eventMessage CONTAINS[c] "safe mode" '
        'OR eventMessage CONTAINS[c] "Previous shutdown cause" '
        'OR eventMessage CONTAINS[c] "Previous shutdown"'
    )
    return [
        run_cmd(
            [
                "log",
                "show",
                "--last",
                f"{hours}h",
                "--style",
                "compact",
                "--predicate",
                predicate,
            ],
            timeout=20,
        )
    ]


def apply_hardening(
    *,
    bin_dir: Path,
    stable_fish: Path,
    dev_fish: Path,
    write_fish: bool,
    write_proxy: bool,
    force: bool,
) -> list[HardenAction]:
    actions: list[HardenAction] = []
    bin_dir.mkdir(parents=True, exist_ok=True)

    if write_fish:
        fish_wrapper = bin_dir / "fish"
        if fish_wrapper.exists() and not force:
            actions.append(
                HardenAction(
                    fish_wrapper,
                    "skipped",
                    "file exists (use --force to overwrite)",
                )
            )
        else:
            content = (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"stable={shlex.quote(str(stable_fish))}\n"
                f"dev={shlex.quote(str(dev_fish))}\n"
                "if [ \"${FISH_ALLOW_DEV:-}\" = \"1\" ] && [ -x \"$dev\" ]; then\n"
                "  exec \"$dev\" \"$@\"\n"
                "fi\n"
                "if [ -x \"$stable\" ]; then\n"
                "  exec \"$stable\" \"$@\"\n"
                "fi\n"
                "if [ -x \"$dev\" ]; then\n"
                "  exec \"$dev\" \"$@\"\n"
                "fi\n"
                "exec /bin/zsh \"$@\"\n"
            )
            write_text(fish_wrapper, content)
            actions.append(
                HardenAction(
                    fish_wrapper,
                    "written",
                    "stable fish wrapper with dev opt-in (FISH_ALLOW_DEV=1)",
                )
            )

        fish_dev = bin_dir / "fish-dev"
        if dev_fish.exists():
            if fish_dev.exists() and not force:
                actions.append(
                    HardenAction(
                        fish_dev,
                        "skipped",
                        "file exists (use --force to overwrite)",
                    )
                )
            else:
                content = (
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    f"exec {shlex.quote(str(dev_fish))} \"$@\"\n"
                )
                write_text(fish_dev, content)
                actions.append(
                    HardenAction(
                        fish_dev,
                        "written",
                        "direct dev fish wrapper",
                    )
                )

    if write_proxy:
        proxy_script = bin_dir / "proxy-off"
        if proxy_script.exists() and not force:
            actions.append(
                HardenAction(
                    proxy_script,
                    "skipped",
                    "file exists (use --force to overwrite)",
                )
            )
        else:
            content = (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "while IFS= read -r service; do\n"
                "  case \"$service\" in\n"
                "    \"\"|\"An asterisk (*) denotes that a network service is disabled.\") continue ;;\n"
                "    \\*) service=${service#\\* } ;;\n"
                "  esac\n"
                "  networksetup -setwebproxystate \"$service\" off >/dev/null 2>&1 || true\n"
                "  networksetup -setsecurewebproxystate \"$service\" off >/dev/null 2>&1 || true\n"
                "  networksetup -setautoproxystate \"$service\" off >/dev/null 2>&1 || true\n"
                "done < <(networksetup -listallnetworkservices)\n"
                "echo \"Proxies disabled for all network services.\"\n"
            )
            write_text(proxy_script, content)
            actions.append(
                HardenAction(
                    proxy_script,
                    "written",
                    "disable web/secure/auto proxies for all services",
                )
            )

    return actions


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.file:
        path = args.file.expanduser().resolve()
        if not path.is_file():
            print(f"error: {path} is not a file", file=sys.stderr)
            return 1
        import anyio

        return anyio.run(analyze_crash_file, path)

    if args.harden:
        actions = apply_hardening(
            bin_dir=args.bin_dir.expanduser(),
            stable_fish=args.stable_fish.expanduser(),
            dev_fish=args.dev_fish.expanduser(),
            write_fish=not args.no_fish_wrapper,
            write_proxy=not args.no_proxy_script,
            force=args.force,
        )
        if actions:
            print("hardening actions")
            for action in actions:
                print(f"- {action.path}: {action.status} ({action.detail})")
        else:
            print("hardening actions: none")
        print("done")
        return 0

    panic_reports = collect_panic_reports(args.max_panics)

    log_results: list[CommandResult] = []
    if not args.no_log:
        log_results = collect_logs(args.hours)

    extra_results: list[CommandResult] = []
    if not args.no_disk:
        extra_results.append(run_cmd(["diskutil", "verifyVolume", "/"], timeout=60))

    if not args.no_proxy:
        extra_results.extend(collect_proxy_settings())

    if not args.no_launch:
        extra_results.extend(collect_launch_items())

    extra_results.extend(collect_shell_info())
    extra_results.extend(collect_recent_reboots())

    report = render_report(
        hours=args.hours,
        max_lines=args.max_lines,
        include_disk=not args.no_disk,
        include_proxy=not args.no_proxy,
        include_launch=not args.no_launch,
        include_logs=not args.no_log,
        panic_reports=panic_reports,
        log_results=log_results,
        extra_results=extra_results,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"wrote report to {args.output}")
        return 0

    sys.stdout.write(report)
    return 0


def entrypoint() -> int:
    return main()
