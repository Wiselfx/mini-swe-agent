import json

import pytest
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.agents.utils.compaction import OUTPUT_REMOVED, context_tokens, is_output, plan
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError
from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output
from tests.agents.test_interactive import mock_prompts


def _step(i: int, text_tokens: int, output_tokens: int) -> list[dict]:
    call = {"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
    return [
        {"role": "assistant", "content": f"step {i} " + "x" * 4 * text_tokens, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": "o" * 4 * output_tokens, "extra": {"returncode": 0}},
    ]


def test_plan_keeps_excerpts_without_outputs_and_whole_tool_calls():
    """First, middle and last excerpts are sized without tool outputs, never split a call from its result, leave the
    rest as gaps (where the user's messages are found), and the recent 30% starts at a message the model can quote."""
    reply = {"role": "user", "content": "Use the other parser.", "extra": {"interrupt_type": "UserReply"}}
    steps = [m for i in range(40) for m in _step(i, text_tokens=200, output_tokens=2_000)]
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        *steps[:30],
        reply,
        *steps[30:],
    ]
    p = plan(messages, first=1_000, middle=1_000, last=2_000, recent=0.3)
    assert p["first"][0] == 2 and p["last"][-1] == len(messages) - 1
    assert p["first"] + p["gap1"] + p["middle"] + p["gap2"] + p["last"] == list(range(2, len(messages)))
    assert all(len(p[k]) > 0 for k in ("gap1", "middle", "gap2"))
    for key in ("first", "middle", "last"):
        assert not is_output(messages[p[key][0]]) and is_output(messages[p[key][-1]])
    assert messages.index(reply) in p["gap1"] + p["gap2"]
    recent = p["recent_start"]
    assert messages[recent]["role"] == "assistant" and 0.6 < recent / len(messages) < 0.8


class _SummarizingModel(DeterministicToolcallModel):
    """Plays scripted steps; a request ending in the compaction instruction gets a summary (a reply without a tool
    call, raised as FormatError with the model's own message first, as the real models do)."""

    def __init__(self, *, context_window: int | None, **kwargs):
        super().__init__(**kwargs)
        self.context_window, self.summary_requests = context_window, []

    def get_context_window(self) -> int:
        return self.context_window

    def query(self, messages, **kwargs):
        if "Pause the task" not in json.dumps(messages[-1]):
            return super().query(messages, **kwargs)
        self.summary_requests.append(messages)
        part = "RECENT" if "recent part" in json.dumps(messages[-1]) else "EARLY"
        raise FormatError(
            {"role": "assistant", "content": f"SUMMARY-{part}"},
            {"role": "user", "content": "No tool calls.", "extra": {"interrupt_type": "FormatError", "cost": 0.25}},
        )


@pytest.mark.parametrize(("context_window", "compact_at", "cached"), [(16_000, 0.87, True), (9_000, 0.97, False)])
def test_compaction_keeps_history_and_summarizes_from_the_cached_conversation(context_window, compact_at, cached):
    """Past 87% of the context window the conversation is compacted once: two summaries (from the cached conversation
    plus one appended request, or from a transcript when there is no room left), tool outputs replaced by stubs, every
    tool call still paired with its result, the full history kept, and the run carries on to submit."""
    config = yaml.safe_load(open("src/minisweagent/config/mini.yaml"))["agent"]
    commands = [f"seq 1 {400 + i}" for i in range(40)] + ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
    outputs = [
        make_toolcall_output(f"step {i}", [{"id": f"c{i}"}], [{"command": c, "tool_call_id": f"c{i}"}])
        for i, c in enumerate(commands)
    ]
    model = _SummarizingModel(outputs=outputs, context_window=context_window)
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **{**config, "compact_at": compact_at, "cost_limit": 0})
    assert agent.run("Count things")["exit_status"] == "Submitted"

    n_compactions = len(agent.compacted_history)
    assert n_compactions >= 1 and len(model.summary_requests) == 2 * n_compactions
    full = agent.compacted_history[0]
    assert len(model.summary_requests[0]) == (len(full) + 1 if cached else 2)
    assert not any(OUTPUT_REMOVED in json.dumps(m) for m in full)
    summary = next(m for m in agent.messages if "<summary>" in str(m["content"]))
    assert "SUMMARY-EARLY" in summary["content"] and "SUMMARY-RECENT" in summary["content"]
    assert agent.messages[:2] == full[:2]
    assert sum(OUTPUT_REMOVED in str(m["content"]) for m in agent.messages) >= 2
    called = [tc["id"] for m in agent.messages if m["role"] == "assistant" for tc in m.get("tool_calls") or []]
    assert called[:-1] == [m["tool_call_id"] for m in agent.messages if m["role"] == "tool"]  # the submit gets none
    assert agent.cost == pytest.approx(len(commands) * 1.0 + 2 * 0.25 * n_compactions)
    assert context_tokens(agent.messages) < compact_at * context_window
    assert agent.serialize()["compacted_history"] == agent.compacted_history


def test_compact_command_runs_before_the_next_model_call():
    """`/compact` typed while a tool call still waits for its result (here at the confirmation prompt) compacts only
    before the next model call, once that result is in, so the summary requests are valid. It also works without a
    known context window, from the cached conversation, and only once."""
    config = yaml.safe_load(open("src/minisweagent/config/mini.yaml"))["agent"]
    commands = [f"seq 1 {50 + i}" for i in range(4)] + ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
    outputs = [
        make_toolcall_output(f"step {i}", [{"id": f"c{i}"}], [{"command": c, "tool_call_id": f"c{i}"}])
        for i, c in enumerate(commands)
    ]
    model = _SummarizingModel(outputs=outputs, context_window=None)
    with mock_prompts(["", "", "/compact", "", "", "", ""]):  # /compact at the 3rd confirmation, then confirm it
        agent = InteractiveAgent(model=model, env=LocalEnvironment(), **{**config, "cost_limit": 0})
        assert agent.run("Count things")["exit_status"] == "Submitted"
    assert len(agent.compacted_history) == 1 and len(model.summary_requests) == 2
    assert len(agent.compacted_history[0]) == 2 + 3 * 2  # system, task and the three steps done when it ran
    assert all(r[-2]["role"] == "tool" for r in model.summary_requests)  # the waiting call had its result by then
    assert len(model.summary_requests[0]) == len(agent.compacted_history[0]) + 1
