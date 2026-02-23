from __future__ import annotations

import subprocess


def get_git_commit() -> str | None:
    """
    Return the current git commit hash if available, else None.
    Works when running inside a git repository.
    """
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None