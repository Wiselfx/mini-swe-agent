"""Pure helpers for compaction: measure the conversation, pick the excerpts to keep, strip tool outputs.

The compacted conversation is (in order): system and task message, the first excerpt, the middle excerpt, the
summaries, the last excerpt. Excerpts are whole tool calls with their results, with the tool outputs removed. User
messages outside the excerpts are kept verbatim, in place, and each dropped stretch leaves a one-line marker.
"""

from itertools import accumulate

from minisweagent.models.utils.content_string import get_content_string

OUTPUT_REMOVED = "[output removed at compaction, run the command again if you need it]"


def estimate_tokens(messages: list[dict]) -> int:
    """Rough count (4 characters per token), for messages no model call has measured yet."""
    return sum(len(get_content_string(m)) // 4 + 4 for m in messages)


def context_tokens(messages: list[dict]) -> int:
    """Size of the next prompt: the last measured prompt and reply, plus an estimate for anything added since."""
    for i in range(len(messages) - 1, -1, -1):
        if usage := messages[i].get("extra", {}).get("usage"):
            return usage["prompt_tokens"] + usage["completion_tokens"] + estimate_tokens(messages[i + 1 :])
    return estimate_tokens(messages)


def is_output(message: dict) -> bool:
    return (
        message.get("role") == "tool"
        or message.get("type") == "function_call_output"
        or "returncode" in message.get("extra", {})
    )


def is_human(message: dict) -> bool:
    """Something the user typed (task, reply, comment, command), as opposed to tool output or agent notices."""
    return (
        message.get("role") == "user"
        and not is_output(message)
        and message.get("extra", {}).get("interrupt_type") not in ("FormatError", "Compaction")
    )


def strip_output(message: dict) -> dict:
    """Copy without its tool output, and without usage, which stops describing the conversation once it changes."""
    message = {
        **message,
        "extra": {k: v for k, v in message.get("extra", {}).items() if k not in ("usage", "raw_output")},
    }
    if is_output(message):
        message["output" if message.get("type") == "function_call_output" else "content"] = OUTPUT_REMOVED
    return message


def _units(messages: list[dict]) -> list[list[int]]:
    """Message indices grouped so a tool call is never separated from its results."""
    units: list[list[int]] = []
    for i, message in enumerate(messages):
        if units and is_output(message) and not is_human(message):
            units[-1].append(i)
        else:
            units.append([i])
    return units


def plan(messages: list[dict], *, first: int, middle: int, last: int, recent: float) -> dict:
    """Pick what to keep from messages[2:] (after system and task message). Excerpt sizes are in tokens without tool
    outputs. Returns index lists for the excerpts and the dropped stretches, plus `recent_start`: the first message of
    the last `recent` share of the conversation, which gets its own detailed summary."""
    units = _units(messages)[2:]
    n = len(units)
    # running totals: kept[k] / full[k] = tokens in units[:k] without / with tool outputs
    kept = [0, *accumulate(estimate_tokens([strip_output(messages[i]) for i in unit]) for unit in units)]
    full = [0, *accumulate(estimate_tokens([messages[i] for i in unit]) for unit in units)]
    n_first = next((k for k in range(n) if kept[k + 1] > first), n)
    n_last = min(next((k for k in range(1, n) if kept[n] - kept[n - k - 1] > last), n) or 1, n - n_first)
    gap = range(n_first, n - n_last)
    center = (kept[n_first] + kept[n - n_last]) / 2
    mid = [k for k in gap if abs((kept[k] + kept[k + 1]) / 2 - center) <= middle / 2]
    recent_unit = next((k for k in range(n) if full[k] >= (1 - recent) * full[n]), n - 1)

    def pick(ks) -> list[int]:
        return [i for k in ks for i in units[k]]

    return {
        "first": pick(range(n_first)),
        "gap1": pick([k for k in gap if not mid or k < mid[0]]),
        "middle": pick(mid),
        "gap2": pick([k for k in gap if mid and k > mid[-1]]),
        "last": pick(range(n - n_last, n)),
        "recent_start": units[recent_unit][0],
    }
