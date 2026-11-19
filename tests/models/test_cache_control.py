import pytest

from minisweagent.models.utils.cache_control import set_cache_control


def _tool_turn(i: int, n_calls: int = 1) -> list[dict]:
    """An assistant message with `n_calls` parallel tool calls, followed by their results."""
    ids = [f"call_{i}_{k}" for k in range(n_calls)]
    calls = [{"id": c, "type": "function", "function": {"name": "bash", "arguments": "{}"}} for c in ids]
    return [
        {"role": "assistant", "content": None, "tool_calls": calls},
        *[{"role": "tool", "tool_call_id": c, "content": f"output {c}"} for c in ids],
    ]


def test_set_cache_control_marks_end_and_previous_end():
    """The last message and the one right before the last assistant turn (where the previous request ended) are
    marked, nothing else. Tool results take the mark on the message and keep their content, and the input is
    left untouched."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Fix the bug."},
        *_tool_turn(1),
        *_tool_turn(2),
    ]
    result = set_cache_control(messages)
    assert [i for i, m in enumerate(result) if m != messages[i]] == [3, 5]
    assert result[3] == {**messages[3], "cache_control": {"type": "ephemeral"}}
    assert result[5] == {**messages[5], "cache_control": {"type": "ephemeral"}}
    assert "cache_control" not in messages[5]


@pytest.mark.parametrize("n_calls", [1, 12])
def test_set_cache_control_repeats_the_previous_request_end(n_calls):
    """The next request sends the previous request's last message exactly as it was sent (same content, same mark),
    however many parallel tool calls came in between, so it can read that request's cache entry."""
    base = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}, *_tool_turn(1)]
    new = [*base, *_tool_turn(2, n_calls)]
    assert set_cache_control(new)[len(base) - 1] == set_cache_control(base)[-1]


def test_set_cache_control_offset_deprecated():
    """Test that last_n_messages_offset parameter has no effect and is deprecated."""
    input_messages = [
        {"role": "user", "content": "First message"},
        {"role": "user", "content": "Second message"},
        {"role": "user", "content": "Third message"},
    ]

    # Test that offset parameter has no effect - should still only affect last message
    result_with_offset = set_cache_control(input_messages, last_n_messages_offset=1)
    result_without_offset = set_cache_control(input_messages)

    # Both results should be identical - offset should have no effect
    assert result_with_offset == result_without_offset

    # Only the last message should have cache control
    expected_output = [
        {"role": "user", "content": "First message"},
        {"role": "user", "content": "Second message"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "Third message", "cache_control": {"type": "ephemeral"}}],
        },
    ]

    assert result_with_offset == expected_output
    assert result_without_offset == expected_output
