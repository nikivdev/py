"""Session management for Claude SDK CLI."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Session:
    """Represents a Claude session."""

    id: str
    project: str
    path: Path
    created_at: datetime | None
    updated_at: datetime | None
    message_count: int
    first_user_message: str | None
    slug: str | None

    @property
    def display_name(self) -> str:
        """Get display name (slug or first message preview)."""
        if self.slug:
            return self.slug
        if self.first_user_message:
            preview = self.first_user_message[:60]
            if len(self.first_user_message) > 60:
                preview += "..."
            return preview
        return self.id[:8]


def get_claude_home() -> Path:
    """Get Claude home directory."""
    return Path.home() / ".claude"


def get_projects_dir() -> Path:
    """Get Claude projects directory."""
    return get_claude_home() / "projects"


def decode_project_path(encoded: str) -> str:
    """Decode project path from directory name."""
    return encoded.replace("-", "/")


def list_projects() -> list[tuple[str, Path]]:
    """List all projects with their paths."""
    projects_dir = get_projects_dir()
    if not projects_dir.exists():
        return []

    projects = []
    for entry in projects_dir.iterdir():
        if entry.is_dir():
            decoded_path = decode_project_path(entry.name)
            projects.append((decoded_path, entry))
    return sorted(projects, key=lambda x: x[0])


def parse_session_file(session_path: Path) -> Session | None:
    """Parse a session JSONL file."""
    if not session_path.exists():
        return None

    session_id = session_path.stem
    project_dir = session_path.parent
    project_path = decode_project_path(project_dir.name)

    created_at = None
    updated_at = None
    message_count = 0
    first_user_message = None
    slug = None

    try:
        with open(session_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Skip file-history-snapshot entries
                if data.get("type") == "file-history-snapshot":
                    continue

                # Extract timestamp
                if "timestamp" in data:
                    ts = data["timestamp"]
                    if isinstance(ts, str):
                        try:
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        except ValueError:
                            dt = None
                    elif isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                    else:
                        dt = None

                    if dt:
                        if created_at is None:
                            created_at = dt
                        updated_at = dt

                # Count messages
                if data.get("type") in ("user", "assistant"):
                    message_count += 1

                # Get first user message
                if data.get("type") == "user" and first_user_message is None:
                    message = data.get("message", {})
                    content = message.get("content", "")
                    if isinstance(content, str):
                        first_user_message = content.strip()
                    elif isinstance(content, list) and content:
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                first_user_message = block.get("text", "").strip()
                                break

                # Get slug
                if "slug" in data and slug is None:
                    slug = data["slug"]

    except Exception:
        return None

    return Session(
        id=session_id,
        project=project_path,
        path=session_path,
        created_at=created_at,
        updated_at=updated_at,
        message_count=message_count,
        first_user_message=first_user_message,
        slug=slug,
    )


def list_sessions(project_filter: str | None = None, limit: int = 50) -> list[Session]:
    """List all sessions, optionally filtered by project."""
    projects_dir = get_projects_dir()
    if not projects_dir.exists():
        return []

    sessions = []
    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue

        decoded_path = decode_project_path(project_dir.name)
        if project_filter and project_filter not in decoded_path:
            continue

        for session_file in project_dir.glob("*.jsonl"):
            # Skip agent sessions for now
            if session_file.stem.startswith("agent-"):
                continue

            session = parse_session_file(session_file)
            if session:
                sessions.append(session)

    # Sort by updated_at (most recent first)
    min_dt = datetime.min.replace(tzinfo=timezone.utc)
    sessions.sort(key=lambda s: s.updated_at or min_dt, reverse=True)
    return sessions[:limit]


def get_session(session_id: str) -> Session | None:
    """Get a session by ID."""
    projects_dir = get_projects_dir()
    if not projects_dir.exists():
        return None

    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue

        session_file = project_dir / f"{session_id}.jsonl"
        if session_file.exists():
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
                if data.get("type") in ("user", "assistant"):
                    messages.append(data)
            except json.JSONDecodeError:
                continue

    return messages
