"""Cache control utilities are mostly for Anthropic models.
They are used to explicitly set cache control points.
"""

import copy
import warnings
from typing import Literal


def _get_content_text(entry: dict) -> str | None:
    if entry["content"] is None:
        return None
    if isinstance(entry["content"], str):
        return entry["content"]
    assert len(entry["content"]) == 1, "Expected single message in content"
    return entry["content"][0]["text"]


def _clear_cache_control(entry: dict) -> None:
    if isinstance(entry["content"], list):
        assert len(entry["content"]) == 1, "Expected single message in content"
        entry["content"][0].pop("cache_control", None)
    # Note: entry["content"] can be None for assistant messages with only tool_use
    entry.pop("cache_control", None)


def _set_cache_control(entry: dict) -> None:
    # None content (assistant messages with only tool_use) and tool results take the mark on the message itself.
    # Tool results keep their content as is, so they look the same whether marked or not and the prefix stays stable.
    if entry["content"] is None or entry["role"] == "tool":
        entry["cache_control"] = {"type": "ephemeral"}
        return

    if not isinstance(entry["content"], list):
        entry["content"] = [  # type: ignore
            {
                "type": "text",
                "text": _get_content_text(entry),
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        entry["content"][0]["cache_control"] = {"type": "ephemeral"}


def set_cache_control(
    messages: list[dict], *, mode: Literal["default_end"] | None = "default_end", last_n_messages_offset: int = 0
) -> list[dict]:
    """This messages processor adds manual cache control marks to the messages."""
    if mode is None:
        return messages
    if mode != "default_end":
        raise ValueError(f"Invalid mode: {mode}")
    if last_n_messages_offset:
        warnings.warn("last_n_messages_offset is deprecated and will be removed in the future. It has no effect.")

    messages = copy.deepcopy(messages)
    # Mark the end, and where the previous request ended (right before the last assistant turn): Anthropic only looks
    # back about 20 blocks for an earlier cache entry, which a turn with many tool calls can exceed.
    previous_end = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=0) - 1
    for i, entry in enumerate(messages):
        _clear_cache_control(entry)
        if i in (previous_end, len(messages) - 1):
            _set_cache_control(entry)
    return messages
