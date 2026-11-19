import re

import pytest

from minisweagent.run.utilities.sessions import SESSIONS_DIR, cwd_dir, new_session_path


@pytest.mark.parametrize(
    ("task", "slug"),
    [
        ("Fix the Prose handling!\nsecond line ignored", "fix-the-prose-handling"),
        ("   ", "session"),
        ("x" * 100, "x" * 40),
        ("ünïcode / only ***", "n-code-only"),
    ],
)
def test_new_session_path(task, slug):
    path = new_session_path(task)
    assert path.parent == cwd_dir() and path.parent.parent == SESSIONS_DIR
    assert re.fullmatch(rf"\d{{8}}-\d{{6}}-{re.escape(slug)}\.traj\.json", path.name)
    assert "/" not in path.parent.name
