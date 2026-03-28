"""Entry point for why-crash."""

from __future__ import annotations

from .cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
