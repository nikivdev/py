"""Session management for Codex SDK CLI."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CodexSession:
    """Represents a Codex session."""

    id: str
    cwd: str
    path: Path
    created_at: datetime | None
    updated_at: datetime | None
    message_count: int
    first_user_message: str | None
    model: str | None
    cli_version: str | None
    source: str | None  # vscode, cli, etc.

    @property
    def display_name(self) -> str:
        """Get display name (first message preview or ID)."""
        if self.first_user_message:
            # Clean up the message - remove context prefixes
            msg = self.first_user_message
            if "## My request for Codex:" in msg:
                msg = msg.split("## My request for Codex:")[-1].strip()
            preview = msg[:60]
            if len(msg) > 60:
                preview += "..."
            return preview
        return self.id[:8]


def get_codex_home() -> Path:
    """Get Codex home directory."""
    return Path.home() / ".codex"


def get_sessions_dir() -> Path:
    """Get Codex sessions directory."""
    return get_codex_home() / "sessions"


def parse_session_file(session_path: Path) -> CodexSession | None:
    """Parse a Codex session JSONL file."""
    if not session_path.exists():
        return None

    # Extract session ID from filename
    # Format: rollout-2025-12-08T12-49-50-019afd94-f997-7da0-be48-a5fc7633baef.jsonl
    filename = session_path.stem
    if filename.startswith("rollout-"):
        # Extract UUID from the end
        parts = filename.split("-")
        if len(parts) >= 6:
            # UUID is the last 5 parts joined
            session_id = "-".join(parts[-5:])
        else:
            session_id = filename
    else:
        session_id = filename

    created_at = None
    updated_at = None
    message_count = 0
    first_user_message = None
    cwd = ""
    model = None
    cli_version = None
    source = None

    try:
        with open(session_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract timestamp
                if "timestamp" in data:
                    ts = data["timestamp"]
                    if isinstance(ts, str):
                        try:
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        except ValueError:
                            dt = None
                    else:
                        dt = None

                    if dt:
                        if created_at is None:
                            created_at = dt
                        updated_at = dt

                # Get session metadata
                if data.get("type") == "session_meta":
                    payload = data.get("payload", {})
                    cwd = payload.get("cwd", "")
                    cli_version = payload.get("cli_version")
                    source = payload.get("source")
                    if "id" in payload:
                        session_id = payload["id"]

                # Get model from turn context
                if data.get("type") == "turn_context":
                    payload = data.get("payload", {})
                    model = payload.get("model")
                    if not cwd:
                        cwd = payload.get("cwd", "")

                # Count user messages and get first one
                if data.get("type") == "event_msg":
                    payload = data.get("payload", {})
                    if payload.get("type") == "user_message":
                        message_count += 1
                        if first_user_message is None:
                            first_user_message = payload.get("message", "").strip()

    except Exception:
        return None

    return CodexSession(
        id=session_id,
        cwd=cwd,
        path=session_path,
        created_at=created_at,
        updated_at=updated_at,
        message_count=message_count,
        first_user_message=first_user_message,
        model=model,
        cli_version=cli_version,
        source=source,
    )


def list_all_session_files() -> list[Path]:
    """Get all session files, traversing the date-based directory structure."""
    sessions_dir = get_sessions_dir()
    if not sessions_dir.exists():
        return []

    session_files = []

    # Handle both flat structure and date-based structure
    for item in sessions_dir.rglob("*.jsonl"):
        if item.is_file():
            session_files.append(item)

    return session_files


def list_sessions(cwd_filter: str | None = None, limit: int = 50) -> list[CodexSession]:
    """List all sessions, optionally filtered by working directory."""
    session_files = list_all_session_files()

    sessions = []
    for session_file in session_files:
        session = parse_session_file(session_file)
        if session:
            if cwd_filter:
                # Normalize paths for comparison
                filter_path = str(Path(cwd_filter).expanduser().resolve())
                if filter_path not in session.cwd and session.cwd not in filter_path:
                    continue
            sessions.append(session)

    # Sort by updated_at (most recent first)
    min_dt = datetime.min.replace(tzinfo=timezone.utc)
    sessions.sort(key=lambda s: s.updated_at or min_dt, reverse=True)
    return sessions[:limit]


def get_session(session_id: str) -> CodexSession | None:
    """Get a session by ID (supports partial match)."""
    session_files = list_all_session_files()

    for session_file in session_files:
        if session_id in session_file.name:
            return parse_session_file(session_file)

    return None


def get_session_messages(session_id: str) -> list[dict]:
    """Get all messages from a session."""
    session = get_session(session_id)
    if not session:
        return []

    messages = []
    with open(session.path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data.get("type") == "event_msg":
                    messages.append(data)
                elif data.get("type") == "response_item":
                    messages.append(data)
            except json.JSONDecodeError:
                continue

    return messages


def get_last_session_for_path(path: str) -> CodexSession | None:
    """Get the most recent session for a given working directory.

    Args:
        path: The project path (can be relative, absolute, or with ~)

    Returns:
        The most recent session for that path, or None if not found
    """
    # Normalize the input path
    resolved_path = str(Path(path).expanduser().resolve())

    sessions = list_sessions(cwd_filter=resolved_path, limit=1)
    return sessions[0] if sessions else None


def list_sessions_for_path(path: str, limit: int = 10) -> list[CodexSession]:
    """List all sessions for a given working directory, sorted by most recent.

    Args:
        path: The project path (can be relative, absolute, or with ~)
        limit: Maximum number of sessions to return

    Returns:
        List of sessions for that path, sorted by updated_at descending
    """
    resolved_path = str(Path(path).expanduser().resolve())
    return list_sessions(cwd_filter=resolved_path, limit=limit)


def list_projects() -> list[tuple[str, int]]:
    """List all unique project paths with session counts."""
    sessions = list_sessions(limit=1000)

    project_counts: dict[str, int] = {}
    for session in sessions:
        if session.cwd:
            project_counts[session.cwd] = project_counts.get(session.cwd, 0) + 1

    return sorted(project_counts.items(), key=lambda x: x[1], reverse=True)
