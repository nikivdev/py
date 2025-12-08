"""Codex SDK CLI - wrapper for OpenAI Codex sessions."""

import typer
from rich.console import Console
from rich.table import Table

from .sessions import (
    list_sessions,
    get_session_messages,
    list_projects,
    get_last_session_for_path,
    list_sessions_for_path,
    get_session,
)

app = typer.Typer(help="Codex SDK CLI - manage OpenAI Codex sessions")
console = Console()


@app.command("sessions")
def sessions_cmd(
    cwd: str | None = typer.Option(None, "-c", "--cwd", help="Filter by working directory"),
    limit: int = typer.Option(20, "-n", "--limit", help="Maximum number of sessions"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show more details"),
):
    """List Codex sessions."""
    sessions = list_sessions(cwd_filter=cwd, limit=limit)

    if not sessions:
        console.print("[yellow]No sessions found[/yellow]")
        return

    table = Table(title="Codex Sessions")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Messages", justify="right")
    table.add_column("Updated", style="yellow")
    if verbose:
        table.add_column("CWD", style="dim")
        table.add_column("Model", style="blue")

    for session in sessions:
        updated = session.updated_at.strftime("%Y-%m-%d %H:%M") if session.updated_at else "-"
        row = [
            session.id[:8],
            session.display_name,
            str(session.message_count),
            updated,
        ]
        if verbose:
            row.append(session.cwd or "-")
            row.append(session.model or "-")
        table.add_row(*row)

    console.print(table)


@app.command("projects")
def projects_cmd():
    """List Codex projects (unique working directories)."""
    projects = list_projects()

    if not projects:
        console.print("[yellow]No projects found[/yellow]")
        return

    table = Table(title="Codex Projects")
    table.add_column("Path", style="cyan")
    table.add_column("Sessions", justify="right")

    for path, count in projects:
        table.add_row(path, str(count))

    console.print(table)


@app.command("show")
def show_cmd(
    session_id: str = typer.Argument(..., help="Session ID (partial match supported)"),
):
    """Show details of a session."""
    session = get_session(session_id)

    if not session:
        console.print(f"[red]No session found matching '{session_id}'[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Session ID:[/bold] {session.id}")
    console.print(f"[bold]CWD:[/bold] {session.cwd}")
    console.print(f"[bold]Messages:[/bold] {session.message_count}")
    if session.model:
        console.print(f"[bold]Model:[/bold] {session.model}")
    if session.source:
        console.print(f"[bold]Source:[/bold] {session.source}")
    if session.cli_version:
        console.print(f"[bold]CLI Version:[/bold] {session.cli_version}")
    if session.created_at:
        console.print(f"[bold]Created:[/bold] {session.created_at}")
    if session.updated_at:
        console.print(f"[bold]Updated:[/bold] {session.updated_at}")

    if session.first_user_message:
        console.print(f"\n[bold]First message:[/bold]")
        console.print(session.first_user_message[:500])


@app.command("messages")
def messages_cmd(
    session_id: str = typer.Argument(..., help="Session ID"),
    limit: int = typer.Option(10, "-n", "--limit", help="Maximum messages to show"),
):
    """Show messages from a session."""
    messages = get_session_messages(session_id)

    if not messages:
        console.print("[yellow]No messages in session[/yellow]")
        return

    # Show last N messages
    for msg in messages[-limit:]:
        msg_type = msg.get("type", "unknown")
        payload = msg.get("payload", {})

        if msg_type == "event_msg":
            if payload.get("type") == "user_message":
                content = payload.get("message", "")
                console.print(f"\n[bold blue]User:[/bold blue]")
                console.print(content[:500])
        elif msg_type == "response_item":
            role = payload.get("role", "")
            if role == "assistant":
                content = payload.get("content", [])
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "output_text":
                            console.print(f"\n[bold green]Assistant:[/bold green]")
                            console.print(block.get("text", "")[:500])


@app.command("last")
def last_cmd(
    path: str = typer.Argument(".", help="Project path (defaults to current directory)"),
    id_only: bool = typer.Option(False, "--id", "-i", help="Only output the session ID (for scripting)"),
    resume_cmd: bool = typer.Option(False, "--resume", "-r", help="Output the codex --resume command"),
):
    """Get the last session for a given path.

    Useful for reconnecting to a session.

    Examples:
        codex-py last ~/projects/myapp
        codex-py last . --id
        codex-py last --resume | pbcopy
    """
    session = get_last_session_for_path(path)

    if not session:
        if not id_only:
            console.print(f"[red]No sessions found for path: {path}[/red]")
        raise typer.Exit(1)

    if id_only:
        print(session.id)
    elif resume_cmd:
        print(f"codex resume {session.id}")
    else:
        console.print(f"[bold]Session ID:[/bold] {session.id}")
        console.print(f"[bold]CWD:[/bold] {session.cwd}")
        console.print(f"[bold]Messages:[/bold] {session.message_count}")
        if session.model:
            console.print(f"[bold]Model:[/bold] {session.model}")
        if session.updated_at:
            console.print(f"[bold]Last updated:[/bold] {session.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if session.display_name:
            console.print(f"[bold]Name:[/bold] {session.display_name}")
        console.print()
        console.print(f"[dim]Resume with:[/dim] codex resume {session.id}")


@app.command("for-path")
def for_path_cmd(
    path: str = typer.Argument(".", help="Project path (defaults to current directory)"),
    limit: int = typer.Option(10, "-n", "--limit", help="Maximum sessions to show"),
):
    """List all sessions for a given path.

    Examples:
        codex-py for-path ~/projects/myapp
        codex-py for-path . -n 5
    """
    sessions = list_sessions_for_path(path, limit=limit)

    if not sessions:
        console.print(f"[yellow]No sessions found for path: {path}[/yellow]")
        return

    table = Table(title=f"Sessions for {path}")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Messages", justify="right")
    table.add_column("Updated", style="yellow")
    table.add_column("Model", style="blue")

    for session in sessions:
        updated = session.updated_at.strftime("%Y-%m-%d %H:%M") if session.updated_at else "-"
        table.add_row(
            session.id[:8],
            session.display_name,
            str(session.message_count),
            updated,
            session.model or "-",
        )

    console.print(table)
    console.print()
    console.print(f"[dim]Resume latest with:[/dim] codex resume {sessions[0].id}")


@app.command("context")
def context_cmd(
    path: str = typer.Argument(".", help="Project path (defaults to current directory)"),
    limit: int = typer.Option(10, "-n", "--limit", help="Number of recent messages to include"),
    copy: bool = typer.Option(True, "--copy/--no-copy", help="Copy to clipboard (default: yes)"),
):
    """Copy the last session's context to clipboard.

    Extracts recent conversation from the last session for the given path
    and copies it to clipboard. Useful for continuing work in a new session.

    Examples:
        codex-py context              # Copy context from last session in current dir
        codex-py context ~/project    # From specific project
        codex-py context -n 5         # Only last 5 messages
        codex-py context --no-copy    # Just print, don't copy
    """
    session = get_last_session_for_path(path)

    if not session:
        console.print(f"[red]No sessions found for path: {path}[/red]")
        raise typer.Exit(1)

    messages = get_session_messages(session.id)

    if not messages:
        console.print("[yellow]No messages in session[/yellow]")
        raise typer.Exit(1)

    # Build context string
    context_parts = []
    context_parts.append(f"# Previous Codex session context")
    context_parts.append(f"# Session: {session.id}")
    context_parts.append(f"# CWD: {session.cwd}")
    if session.updated_at:
        context_parts.append(f"# Last updated: {session.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    context_parts.append("")

    # Get last N messages
    recent_messages = messages[-limit:]
    for msg in recent_messages:
        msg_type = msg.get("type", "unknown")
        payload = msg.get("payload", {})

        if msg_type == "event_msg":
            if payload.get("type") == "user_message":
                content = payload.get("message", "")
                context_parts.append("## User:")
                context_parts.append(content.strip())
                context_parts.append("")
        elif msg_type == "response_item":
            role = payload.get("role", "")
            if role == "assistant":
                content_blocks = payload.get("content", [])
                for block in content_blocks:
                    if isinstance(block, dict):
                        if block.get("type") == "output_text":
                            context_parts.append("## Assistant:")
                            context_parts.append(block.get("text", "").strip())
                            context_parts.append("")

    context_str = "\n".join(context_parts)

    if copy:
        try:
            import subprocess
            process = subprocess.Popen(
                ["pbcopy"],
                stdin=subprocess.PIPE,
            )
            process.communicate(context_str.encode("utf-8"))
            console.print(f"[green]Copied {len(recent_messages)} messages to clipboard[/green]")
            console.print(f"[dim]From session: {session.id[:8]}[/dim]")
        except Exception as e:
            console.print(f"[red]Failed to copy to clipboard: {e}[/red]")
            console.print(context_str)
    else:
        console.print(context_str)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
