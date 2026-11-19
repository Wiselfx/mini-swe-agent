"""Session files for `mini`: one trajectory per session, grouped by working directory.

Layout: `<global_config_dir>/sessions/<cwd-slug>/<YYYYMMDD-HHMMSS>-<task-slug>.traj.json`.
Listing never opens the files, so it stays fast with thousands of sessions.
"""

import re
import time
from datetime import datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, FuzzyCompleter, WordCompleter
from prompt_toolkit.document import Document

from minisweagent import global_config_dir

SESSIONS_DIR = global_config_dir / "sessions"
SUFFIX = ".traj.json"


def cwd_dir(cwd: Path | None = None) -> Path:
    return SESSIONS_DIR / re.sub(r"[^A-Za-z0-9._-]", "-", str((cwd or Path.cwd()).resolve()))


def new_session_path(task: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", (task.strip().splitlines() or [""])[0].lower()).strip("-")[:40].strip("-")
    return cwd_dir() / f"{datetime.now():%Y%m%d-%H%M%S}-{slug or 'session'}{SUFFIX}"


def list_sessions() -> list[Path]:
    """All sessions, current directory first, each group most recently active first."""
    here = cwd_dir()
    return sorted(SESSIONS_DIR.glob(f"*/*{SUFFIX}"), key=lambda p: (p.parent != here, -p.stat().st_mtime))


def _ago(path: Path) -> str:
    seconds = time.time() - path.stat().st_mtime
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} ago"
    return "just now"


def pick_session() -> Path:
    """Fuzzy-pick a session. Enter takes the highlighted or best-matching row, or the most recent one."""
    sessions = list_sessions()
    if not sessions:
        raise FileNotFoundError(f"No sessions found in {SESSIONS_DIR}")
    here = cwd_dir()
    keys = {f"{p.parent.name}/{p.name.removesuffix(SUFFIX)}": p for p in sessions}
    display = {k: k.split("/", 1)[1] if p.parent == here else k for k, p in keys.items()}
    completer = FuzzyCompleter(
        WordCompleter(list(keys), display_dict=display, meta_dict={k: _ago(p) for k, p in keys.items()}, sentence=True),
        WORD=True,
    )
    print(
        f"{len(sessions)} sessions ({sum(p.parent == here for p in sessions)} in this directory). "
        "Type to filter, arrows to choose, Enter to resume."
    )
    session = PromptSession(completer=completer, complete_while_typing=True)
    answer = session.prompt("resume> ", pre_run=lambda: session.default_buffer.start_completion()).strip()
    if not answer:
        return sessions[0]
    if answer in keys:
        return keys[answer]
    if best := next(iter(completer.get_completions(Document(answer), CompleteEvent())), None):
        return keys[best.text]
    raise ValueError(f"No session matches {answer!r}")
