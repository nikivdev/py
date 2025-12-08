"""Claude SDK CLI - wrapper over claude-agent-sdk."""

import sys

import typer
from rich.console import Console
from rich.table import Table

from .sessions import list_sessions, get_session_messages, list_projects, get_last_session_for_path, list_sessions_for_path
from .run import run_task_sync

app = typer.Typer(help="Claude SDK CLI - manage Claude sessions and more")
console = Console()


@app.command("sessions")
def sessions_cmd(
    project: str | None = typer.Option(None, "-p", "--project", help="Filter by project path"),
    limit: int = typer.Option(20, "-n", "--limit", help="Maximum number of sessions"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show more details"),
):
    """List Claude sessions."""
    sessions = list_sessions(project_filter=project, limit=limit)

    if not sessions:
        console.print("[yellow]No sessions found[/yellow]")
        return

    table = Table(title="Claude Sessions")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Messages", justify="right")
    table.add_column("Updated", style="yellow")
    if verbose:
        table.add_column("Project", style="dim")

    for session in sessions:
        updated = session.updated_at.strftime("%Y-%m-%d %H:%M") if session.updated_at else "-"
        row = [
            session.id[:8],
            session.display_name,
            str(session.message_count),
            updated,
        ]
        if verbose:
            row.append(session.project)
        table.add_row(*row)

    console.print(table)


@app.command("projects")
def projects_cmd():
    """List Claude projects."""
    projects = list_projects()

    if not projects:
        console.print("[yellow]No projects found[/yellow]")
        return

    table = Table(title="Claude Projects")
    table.add_column("Path", style="cyan")
    table.add_column("Sessions", justify="right")

    for path, dir_path in projects:
        session_count = len(list(dir_path.glob("*.jsonl")))
        table.add_row(path, str(session_count))

    console.print(table)


@app.command("show")
def show_cmd(
    session_id: str = typer.Argument(..., help="Session ID (partial match supported)"),
):
    """Show details of a session."""
    # Support partial session ID matching
    sessions = list_sessions(limit=1000)
    matching = [s for s in sessions if s.id.startswith(session_id)]

    if not matching:
        console.print(f"[red]No session found matching '{session_id}'[/red]")
        raise typer.Exit(1)

    if len(matching) > 1:
        console.print(f"[yellow]Multiple sessions match '{session_id}':[/yellow]")
        for s in matching[:10]:
            console.print(f"  {s.id}")
        raise typer.Exit(1)

    session = matching[0]
    console.print(f"[bold]Session:[/bold] {session.id}")
    console.print(f"[bold]Project:[/bold] {session.project}")
    console.print(f"[bold]Messages:[/bold] {session.message_count}")
    if session.slug:
        console.print(f"[bold]Slug:[/bold] {session.slug}")
    if session.created_at:
        console.print(f"[bold]Created:[/bold] {session.created_at}")
    if session.updated_at:
        console.print(f"[bold]Updated:[/bold] {session.updated_at}")

    # Show first user message
    if session.first_user_message:
        console.print(f"\n[bold]First message:[/bold]")
        console.print(session.first_user_message[:500])


@app.command("messages")
def messages_cmd(
    session_id: str = typer.Argument(..., help="Session ID"),
    limit: int = typer.Option(10, "-n", "--limit", help="Maximum messages to show"),
):
    """Show messages from a session."""
    sessions = list_sessions(limit=1000)
    matching = [s for s in sessions if s.id.startswith(session_id)]

    if not matching:
        console.print(f"[red]No session found matching '{session_id}'[/red]")
        raise typer.Exit(1)

    session = matching[0]
    messages = get_session_messages(session.id)

    if not messages:
        console.print("[yellow]No messages in session[/yellow]")
        return

    # Show last N messages
    for msg in messages[-limit:]:
        msg_type = msg.get("type", "unknown")
        message_data = msg.get("message", {})

        if msg_type == "user":
            content = message_data.get("content", "")
            if isinstance(content, str):
                console.print(f"\n[bold blue]User:[/bold blue]")
                console.print(content[:500])
        elif msg_type == "assistant":
            content = message_data.get("content", [])
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        console.print(f"\n[bold green]Assistant:[/bold green]")
                        console.print(block.get("text", "")[:500])
                    elif block.get("type") == "tool_use":
                        console.print(f"\n[bold yellow]Tool:[/bold yellow] {block.get('name', 'unknown')}")


@app.command("run")
def run_cmd(
    path: str = typer.Argument(..., help="Working directory path"),
    task: str = typer.Argument(..., help="Task description for Claude"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Print progress output"),
):
    """Run a task autonomously in the given path.

    Exits with code 0 on success, 1 on error.
    Silent by default - only prints on error unless --verbose.
    """
    exit_code = run_task_sync(path, task, verbose)
    raise typer.Exit(exit_code)


@app.command("last")
def last_cmd(
    path: str = typer.Argument(".", help="Project path (defaults to current directory)"),
    id_only: bool = typer.Option(False, "--id", "-i", help="Only output the session ID (for scripting)"),
    resume_cmd: bool = typer.Option(False, "--resume", "-r", help="Output the claude --resume command"),
):
    """Get the last session for a given path.

    Useful for reconnecting to a session after closing Cursor.

    Examples:
        claude-sdk last ~/projects/myapp
        claude-sdk last . --id
        claude-sdk last --resume | pbcopy
    """
    session = get_last_session_for_path(path)

    if not session:
        if not id_only:
            console.print(f"[red]No sessions found for path: {path}[/red]")
        raise typer.Exit(1)

    if id_only:
        # Just print the ID for scripting
        print(session.id)
    elif resume_cmd:
        # Print the command to resume this session
        print(f"claude --resume {session.id}")
    else:
        # Pretty output
        console.print(f"[bold]Session ID:[/bold] {session.id}")
        console.print(f"[bold]Project:[/bold] {session.project}")
        console.print(f"[bold]Messages:[/bold] {session.message_count}")
        if session.updated_at:
            console.print(f"[bold]Last updated:[/bold] {session.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if session.display_name:
            console.print(f"[bold]Name:[/bold] {session.display_name}")
        console.print()
        console.print(f"[dim]Resume with:[/dim] claude --resume {session.id}")


@app.command("for-path")
def for_path_cmd(
    path: str = typer.Argument(".", help="Project path (defaults to current directory)"),
    limit: int = typer.Option(10, "-n", "--limit", help="Maximum sessions to show"),
):
    """List all sessions for a given path.

    Examples:
        claude-sdk for-path ~/projects/myapp
        claude-sdk for-path . -n 5
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

    for session in sessions:
        updated = session.updated_at.strftime("%Y-%m-%d %H:%M") if session.updated_at else "-"
        table.add_row(
            session.id[:8],
            session.display_name,
            str(session.message_count),
            updated,
        )

    console.print(table)
    console.print()
    console.print(f"[dim]Resume latest with:[/dim] claude --resume {sessions[0].id}")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
