"""Run autonomous Claude tasks."""

import sys
from pathlib import Path

import anyio

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)


async def run_task(path: str, task: str, verbose: bool = False) -> int:
    """Run a task autonomously in the given path.

    Args:
        path: Working directory for the task
        task: Task description/prompt for Claude
        verbose: Print progress output

    Returns:
        0 on success, 1 on error
    """
    cwd = Path(path).expanduser().resolve()
    if not cwd.exists():
        print(f"Error: Path does not exist: {cwd}", file=sys.stderr)
        return 1

    if not cwd.is_dir():
        print(f"Error: Path is not a directory: {cwd}", file=sys.stderr)
        return 1

    options = ClaudeAgentOptions(
        cwd=str(cwd),
        permission_mode="bypassPermissions",
        max_turns=50,
    )

    is_error = False
    error_message = None

    try:
        async for message in query(prompt=task, options=options):
            if isinstance(message, AssistantMessage):
                if verbose:
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(block.text)
                if message.error:
                    is_error = True
                    error_message = message.error
            elif isinstance(message, ResultMessage):
                if message.is_error:
                    is_error = True
                    if message.result:
                        error_message = message.result
                if verbose:
                    print(f"\n--- Completed in {message.duration_ms}ms ---")
                    if message.total_cost_usd:
                        print(f"Cost: ${message.total_cost_usd:.4f}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if is_error:
        if error_message:
            print(f"Task failed: {error_message}", file=sys.stderr)
        return 1

    return 0


def run_task_sync(path: str, task: str, verbose: bool = False) -> int:
    """Synchronous wrapper for run_task."""
    return anyio.run(run_task, path, task, verbose)
