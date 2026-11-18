"""A small generalization of the default agent that puts the user in the loop.

There are three modes:
- human: commands issued by the user are executed immediately
- confirm: commands issued by the LM but not whitelisted are confirmed by the user
- yolo: commands issued by the LM are executed immediately without confirmation
"""

import os
import re
import sys
from typing import Literal, NoReturn

from rich.console import Console
from rich.markup import escape
from rich.rule import Rule

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.agents.utils.prompt_user import _multiline_prompt, prompt_session
from minisweagent.exceptions import LimitsExceeded, Submitted, TimeExceeded, UserInterruption
from minisweagent.models.utils.content_string import get_content_string

console = Console(highlight=False)


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean-ish environment variable ("0", "false", "no", "off" are falsy)."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


_MAX_PREVIEW_LINE_CHARS = 200


def _shorten(line: str, limit: int = _MAX_PREVIEW_LINE_CHARS) -> str:
    """Keep one pathological line (a minified bundle, a 500k-char JSON blob) from filling the screen."""
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _get_text(message: dict) -> str:
    """The message's plain reply text, without tool calls folded in (get_content_string does that)."""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in (None, "text")
        ]
        return "\n\n".join(p for p in parts if p).strip()
    return ""


_REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")



def _get_reasoning(message: dict) -> str:
    """Pull the model's thinking out of whatever field the provider used.

    litellm normalizes most providers to ``reasoning_content``, but ``reasoning``
    and ``thinking`` show up too.
    """
    for key in _REASONING_KEYS:
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):  # e.g. [{"type": "thinking", "thinking": "..."}]
            parts = [v.get("thinking") or v.get("text") or "" for v in value if isinstance(v, dict)]
            if joined := "".join(parts).strip():
                return joined
        if isinstance(value, dict) and (text := (value.get("text") or value.get("content") or "").strip()):
            return text
    return ""


class InteractiveAgentConfig(AgentConfig):
    mode: Literal["human", "confirm", "yolo"] = "confirm"
    """Whether to confirm actions."""
    whitelist_actions: list[str] = []
    """Never confirm actions that match these regular expressions."""
    confirm_exit: bool = True
    """If the agent wants to finish, do we ask for confirmation from user?"""
    show_observations: bool = not _env_flag("MSWEA_QUIET")
    """Print the full tool output. When False, print a head/tail preview instead (MSWEA_QUIET=1)."""
    observation_preview_lines: int = 4
    """Lines shown from the top and from the bottom of a truncated tool output."""


def _fmt_cost(cost: float) -> str:
    """Keep sub-cent spend visible instead of rounding it to $0.0000."""
    if cost == 0:
        return "$0.00"
    if cost < 0.01:
        return f"${cost:.6f}"
    if cost < 1:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _fmt_tokens(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}k"
    return str(count)


class InteractiveAgent(DefaultAgent):
    _MODE_COMMANDS_MAPPING = {"/u": "human", "/c": "confirm", "/y": "yolo"}
    # Cycle order for /e. "none" disables thinking entirely; the rest are passed as
    # `reasoning_effort` (levels servers do not implement are ignored or dropped).
    _THINKING_LEVELS = ("high", "medium", "low", "minimal", "none")

    def __init__(self, *args, config_class=InteractiveAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.cost_last_confirmed = 0.0

    def _interrupt(self, content: str, *, itype: str = "UserInterruption") -> NoReturn:
        raise UserInterruption({"role": "user", "content": content, "extra": {"interrupt_type": itype}})

    def _print_tool_calls(self, message: dict) -> None:
        """Print which tool was called and with what arguments (never the result)."""
        if self.config.mode == "confirm":
            # The confirmation prompt lists the commands that are about to run, no need to repeat them
            return
        for action in message.get("extra", {}).get("actions", []):
            command = action.get("command", "")
            if "\n" in command.strip():
                console.print(f"[cyan]bash[/cyan][dim] ($)[/dim]")
                console.print(_indent(command.rstrip()), highlight=False, markup=False, style="dim")
            else:
                console.print(
                    f"[cyan]bash[/cyan][dim] ($)[/dim] {escape(command)}", highlight=False
                )

    def _print_observation(self, message: dict) -> None:
        """Print the exit code plus a head/tail preview of the command output."""
        extra = message.get("extra", {})
        returncode = extra.get("returncode")
        outcome = "not executed" if returncode in (None, -1) else f"exit {returncode}"
        console.print(f"[dim]↳ {outcome}[/dim]", highlight=False)

        output = extra.get("raw_output")
        if not isinstance(output, str):
            output = get_content_string(message)
        output = output.strip("\n")
        if not output.strip():
            return

        limit = max(1, self.config.observation_preview_lines)
        width = max(20, min(_MAX_PREVIEW_LINE_CHARS, console.width - 2))
        lines = [_shorten(line, width) for line in output.splitlines()]
        if len(lines) <= 2 * limit:
            visible, elided = lines, 0
        else:
            visible = lines[:limit] + [None] + lines[-limit:]
            elided = len(lines) - 2 * limit

        for line in visible:
            if line is None:
                console.print(f"[dim]  … {elided} lines elided …[/dim]", highlight=False)
            else:
                console.print(f"  {line}", highlight=False, markup=False)

    def add_messages(self, *messages: dict) -> list[dict]:
        # Extend supermethod to print messages
        for msg in messages:
            role, content = msg.get("role") or msg.get("type", "unknown"), get_content_string(msg)
            if role == "tool":
                if not self.config.show_observations:
                    self._print_observation(msg)
                    continue
            elif role == "assistant":
                console.print(
                    f"\n[red][bold]mini-swe-agent[/bold]:[/red]\n",
                    end="",
                    highlight=False,
                )
                if reasoning := _get_reasoning(msg):
                    console.print(reasoning, style="dim italic", highlight=False, markup=False)
                if not self.config.show_observations:
                    # Print the reply text, then an explicit "tool + command" line instead of the raw block
                    if text := _get_text(msg):
                        console.print(text, highlight=False, markup=False)
                    self._print_tool_calls(msg)
                    continue
            else:
                console.print(f"\n[bold green]{role.capitalize()}[/bold green]:\n", end="", highlight=False)
            if content.strip():  # e.g. the final "exit" message carries no text
                console.print(content, highlight=False, markup=False)
        return super().add_messages(*messages)
        return super().add_messages(*messages)

    def query(self) -> dict:
        # Extend supermethod to handle human mode
        if self.config.mode == "human":
            match command := self._prompt_and_handle_slash_commands("[bold yellow]>[/bold yellow] "):
                case "/y" | "/c":
                    pass
                case _:
                    msg = {
                        "role": "user",
                        "content": f"User command: \n```bash\n{command}\n```",
                        "extra": {"actions": [{"command": command}]},
                    }
                    self.add_messages(msg)
                    return msg
        try:
            with console.status("Waiting for the LM to respond..."):
                return super().query()
        except TimeExceeded:
            # A wall-clock limit can't be lifted by raising the step/cost limits
            # (the next query re-checks the clock and raises again), so prompting
            # would loop forever. Always stop cleanly instead.
            raise
        except LimitsExceeded:
            if not self._stdin_is_interactive():
                # No terminal to prompt for new limits -- e.g. an unattended
                # `--yolo` run, or any run with stdin redirected from /dev/null
                # (CI, a sandbox). Stop cleanly so the trajectory is saved with a
                # LimitsExceeded exit status, instead of crashing on EOFError when
                # reading input.
                raise
            console.print(
                f"Limits exceeded. Limits: {self.config.step_limit} steps, ${self.config.cost_limit}.\n"
                f"Current spend: {self.n_calls} steps, ${self.cost:.2f}."
            )
            self.config.step_limit = int(input("New step limit: "))
            self.config.cost_limit = float(input("New cost limit: "))
            return super().query()

    @staticmethod
    def _stdin_is_interactive() -> bool:
        """Whether an interactive terminal is available to prompt the user.

        Returns False for unattended runs (e.g. `--yolo` in CI, or inside a
        sandbox with stdin redirected from /dev/null), where calling ``input()``
        would raise ``EOFError`` and crash the run.
        """
        try:
            return sys.stdin is not None and sys.stdin.isatty()
        except (ValueError, OSError):
            return False

    def step(self) -> list[dict]:
        # Override the step method to handle user interruption
        console.print(Rule())
        result: list[dict] = []
        try:
            result = super().step()
        except KeyboardInterrupt:
            interruption_message = self._prompt_and_handle_slash_commands(
                "\n\n[bold yellow]Interrupted.[/bold yellow] "
                "[green]Type a comment/command[/green] (/h for available commands)\n"
                "[bold yellow]>[/bold yellow] "
            ).strip()
            if not interruption_message or interruption_message in self._MODE_COMMANDS_MAPPING:
                interruption_message = "Temporary interruption caught."
            self._interrupt(f"Interrupted by user: {interruption_message}")
        finally:
            if self.messages:
                self._print_footer()
        return result

    def execute_actions(self, message: dict) -> list[dict]:
        # Override to handle user confirmation and confirm_exit, with try/finally to preserve partial outputs
        actions = message.get("extra", {}).get("actions", [])
        commands = [action["command"] for action in actions]
        outputs = []
        try:
            self._ask_confirmation_or_interrupt(commands)
            for action in actions:
                outputs.append(self.env.execute(action))
        except Submitted as e:
            self._check_for_new_task_or_submit(e)
        finally:
            result = self.add_messages(
                *self.model.format_observation_messages(message, outputs, self.get_template_vars())
            )
        return result

    def _add_observation_messages(self, message: dict, outputs: list[dict]) -> list[dict]:
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def _check_for_new_task_or_submit(self, e: Submitted) -> NoReturn:
        """Check if user wants to add a new task or submit."""
        if self.config.confirm_exit:
            message = (
                "[bold yellow]Agent wants to finish.[/bold yellow] "
                "[bold green]Type new task[/bold green] or [bold]Enter[/bold] to quit "
                "([bold]/h[/bold] for commands)\n"
                "[bold yellow]>[/bold yellow] "
            )
            user_input = self._prompt_and_handle_slash_commands(message).strip()
            if user_input == "/u":  # directly continue
                self._interrupt("Switched to human mode.")
            elif user_input in self._MODE_COMMANDS_MAPPING:  # ask again
                return self._check_for_new_task_or_submit(e)
            elif user_input:
                self._interrupt(f"The user added a new task: {user_input}", itype="UserNewTask")
        raise e

    def _should_ask_confirmation(self, action: str) -> bool:
        return self.config.mode == "confirm" and not any(re.match(r, action) for r in self.config.whitelist_actions)

    def _ask_confirmation_or_interrupt(self, commands: list[str]) -> None:
        if not any(self._should_ask_confirmation(c) for c in commands):
            return
        console.print(f"\n[bold yellow]Agent wants to run {len(commands)} action(s):[/bold yellow]")
        for command in commands:
            if "\n" in command.strip():
                console.print(_indent(command.rstrip()), highlight=False, markup=False, style="cyan")
            else:
                console.print(f"  [cyan]{escape(command)}[/cyan]", highlight=False)
        prompt = (
            f"[bold yellow]Execute {len(commands)} action(s)?[/] [green][bold]Enter[/] to confirm[/], "
            "[red]type [bold]comment[/] to reject[/], or [blue][bold]/h[/] to show available commands[/]\n"
            "[bold yellow]>[/bold yellow] "
        )
        match user_input := self._prompt_and_handle_slash_commands(prompt).strip():
            case "" | "/y":
                pass  # confirmed, do nothing
            case "/u":  # Skip execution action and get back to query
                self._interrupt("Commands not executed. Switching to human mode", itype="UserRejection")
            case _:
                self._interrupt(
                    f"Commands not executed. The user rejected your commands with the following message: {user_input}",
                    itype="UserRejection",
                )

    def _thinking_levels(self) -> list[str]:
        """Levels the active model accepts, from the model when it can report them."""
        if callable(getattr(self.model, "get_thinking_levels", None)):
            levels = self.model.get_thinking_levels()
            if levels is not None:
                return list(levels)
        return list(self._THINKING_LEVELS)

    def _thinking_level(self) -> str:
        """Current `reasoning_effort` sent to the model ("unset" if the model has no model_kwargs)."""
        try:
            return str(self.model.config.model_kwargs.get("reasoning_effort", "unset"))
        except AttributeError:
            return "unset"

    def _set_thinking_level(self, level: str) -> None:
        try:
            self.model.config.model_kwargs["reasoning_effort"] = level
        except (AttributeError, TypeError, ValueError):
            console.print(
                "[bold red]This model class does not expose model_kwargs, cannot switch thinking level.[/bold red]"
            )
            return
        console.print(f"Thinking level: [bold green]{level}[/bold green]")

    def _cycle_thinking_level(self) -> None:
        levels = self._thinking_levels()
        if not levels:
            console.print("[bold yellow]This model accepts no reasoning level.[/bold yellow]")
            return
        current = self._thinking_level()
        next_level = levels[(levels.index(current) + 1) % len(levels)] if current in levels else levels[0]
        self._set_thinking_level(next_level)

    def preflight(self) -> None:
        """Pick model and thinking level before the task prompt. Enter keeps the current value."""
        if self._model_profiles():
            self._print_model_picker()
            choice = prompt_session.prompt("Model (number/name, Enter to keep): ").strip()
            if choice:
                self._handle_model_command(choice)
        else:
            console.print(f"Model: [bold green]{escape(self._current_model_name())}[/bold green]")

        levels = self._thinking_levels()
        if not levels:
            console.print("[yellow]This model accepts no reasoning level, skipping.[/yellow]")
            return
        current = self._thinking_level()
        listing = "  ".join(
            f"[bold green]{i}[/bold green]. {level}" + (" [dim](current)[/dim]" if level == current else "")
            for i, level in enumerate(levels, 1)
        )
        console.print(f"Thinking level:\n{listing}")
        choice = prompt_session.prompt("Thinking (number/name, Enter to keep): ").strip()
        if not choice:
            return
        if choice.isdigit() and 1 <= int(choice) <= len(levels):
            choice = levels[int(choice) - 1]
        if choice.lower() in levels:
            self._set_thinking_level(choice.lower())
        else:
            console.print(
                f"[bold red]Unknown thinking level '{escape(choice)}'.[/bold red] "
                f"Choose one of: {', '.join(levels)}"
            )

    def _total_cost(self) -> float:
        """Sum the cost recorded on each message.

        `self.cost` is only charged for a failed call once `run()` handles the FormatError, which
        happens after the step's footer is printed, so summing the messages keeps it current.
        """
        recorded = sum(
            float((message.get("extra") or {}).get("cost") or 0.0)
            for message in self.messages
            if isinstance(message.get("extra"), dict)
        )
        return max(recorded, self.cost)

    def _usage_totals(self) -> dict[str, int]:
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cache_write_tokens": 0}
        context_tokens = 0
        for message in self.messages:
            usage = (message.get("extra") or {}).get("usage")
            if not isinstance(usage, dict):
                continue
            for key in totals:
                totals[key] += int(usage.get(key) or 0)
            if usage.get("prompt_tokens"):
                context_tokens = int(usage["prompt_tokens"]) + int(usage.get("completion_tokens") or 0)
        totals["context_tokens"] = context_tokens
        return totals

    def _active_profile_name(self) -> str | None:
        model_name = self._current_model_name()
        for name, profile in self._model_profiles().items():
            if profile.get("model_name") == model_name:
                return name
        return None

    def _footer_segments(self) -> list[str]:
        usage = self._usage_totals()
        label = self._active_profile_name() or self._current_model_name()
        segments = [label, f"step {self.n_calls}", _fmt_cost(self._total_cost())]

        window = 0
        if callable(getattr(self.model, "get_context_window", None)):
            window = self.model.get_context_window()
        window = window if isinstance(window, int) else 0
        if usage["context_tokens"]:
            if window:
                pct = 100 * usage["context_tokens"] / window
                segments.append(f"ctx {_fmt_tokens(usage['context_tokens'])}/{_fmt_tokens(window)} {pct:.0f}%")
            else:
                segments.append(f"ctx {_fmt_tokens(usage['context_tokens'])}")

        if usage["prompt_tokens"]:
            hit = 100 * usage["cached_tokens"] / usage["prompt_tokens"]
            segments.append(f"cache {_fmt_tokens(usage['cached_tokens'])} read, {hit:.0f}% hit")
            if usage["cache_write_tokens"]:
                segments.append(f"{_fmt_tokens(usage['cache_write_tokens'])} written")
        return segments

    def _print_footer(self) -> None:
        segments = self._footer_segments()
        # Skip repeated footers (e.g. a step that aborted on a limit before doing any work)
        signature = (tuple(segments), len(self.messages))
        if signature == getattr(self, "_last_footer", None):
            return
        self._last_footer = signature
        console.print(f"[dim]{escape('  │  '.join(segments))}[/dim]", highlight=False)

    def _model_profiles(self) -> dict[str, dict]:
        profiles = getattr(getattr(self.model, "config", None), "model_profiles", None)
        return dict(profiles) if isinstance(profiles, dict) else {}

    def _current_model_name(self) -> str:
        return str(getattr(getattr(self.model, "config", None), "model_name", "unknown"))

    def _print_model_picker(self) -> None:
        profiles = self._model_profiles()
        if not profiles:
            console.print(
                "[bold yellow]No model profiles configured.[/bold yellow] Add `model_profiles` "
                "to your config file, then run /model again."
            )
            return
        current = self._current_model_name()
        col = max(len(name) for name in profiles) + 2
        room = max(20, console.width - col - 20)
        lines = []
        for i, (name, profile) in enumerate(profiles.items(), 1):
            model_name = _shorten(str(profile.get("model_name", current)), room)
            marker = " [dim](current)[/dim]" if profile.get("model_name", current) == current else ""
            lines.append(f"[bold green]{i}[/bold green]. {name.ljust(col)}[dim]{escape(model_name)}[/dim]{marker}")
        console.print("Available models:\n" + "\n".join(lines))

    def _apply_model_profile(self, name: str) -> None:
        profiles = self._model_profiles()
        if name not in profiles:
            console.print(
                f"[bold red]Unknown model profile '{escape(name)}'.[/bold red] "
                f"Choose one of: {', '.join(profiles) or '(none configured)'}"
            )
            return
        try:
            model_name = self.model.apply_profile(name)
        except (AttributeError, KeyError, ValueError) as e:
            console.print(f"[bold red]Could not switch model: {escape(str(e))}[/bold red]")
            return
        console.print(f"Model: [bold green]{escape(name)}[/bold green] [dim]({escape(str(model_name))})[/dim]")

    def _handle_model_command(self, argument: str) -> None:
        profiles = self._model_profiles()
        if not profiles:
            self._print_model_picker()
            return
        argument = argument.strip()
        if not argument:
            self._print_model_picker()
            argument = prompt_session.prompt("Number or name (Enter to cancel): ").strip()
            if not argument:
                return
        if argument.isdigit() and 1 <= int(argument) <= len(profiles):
            argument = list(profiles)[int(argument) - 1]
        matched = [name for name in profiles if name.lower() == argument.lower()]
        if not matched:
            matched = [name for name in profiles if name.lower().startswith(argument.lower())]
        if len(matched) != 1:
            self._apply_model_profile(argument)
            return
        self._apply_model_profile(matched[0])

    def _prompt_and_handle_slash_commands(self, prompt: str, *, _multiline: bool = False) -> str:
        """Prompts the user, takes care of /h (followed by requery) and sets the mode. Returns the user input."""
        console.print(prompt, end="")
        if _multiline:
            return _multiline_prompt()
        user_input = prompt_session.prompt("")
        if user_input == "/m":
            return self._prompt_and_handle_slash_commands(prompt, _multiline=True)
        if user_input == "/h":
            console.print(
                f"Current mode: [bold green]{self.config.mode}[/bold green]\n"
                f"[bold green]/y[/bold green] to switch to [bold yellow]yolo[/bold yellow] mode (execute LM commands without confirmation)\n"
                f"[bold green]/c[/bold green] to switch to [bold yellow]confirmation[/bold yellow] mode (ask for confirmation before executing LM commands)\n"
                f"[bold green]/u[/bold green] to switch to [bold yellow]human[/bold yellow] mode (execute commands issued by the user)\n"
                f"[bold green]/m[/bold green] to enter multiline comment\n"
                f"[bold green]/e[/bold green] to cycle the thinking level (currently [bold green]{self._thinking_level()}[/bold green]: "
                f"{' -> '.join(self._thinking_levels()) or 'none supported'})\n"
                f"[bold green]/e <level>[/bold green] to set it directly (e.g. [bold green]/e low[/bold green])\n"
                f"[bold green]/model[/bold green] to pick another model from `model_profiles` "
                f"(currently [bold green]{escape(self._current_model_name())}[/bold green])",
            )
            return self._prompt_and_handle_slash_commands(prompt)
        if user_input == "/e" or user_input.startswith("/e "):
            level = user_input[2:].strip().lower()
            levels = self._thinking_levels()
            if not level:
                self._cycle_thinking_level()
            elif level in levels:
                self._set_thinking_level(level)
            else:
                console.print(
                    f"[bold red]Unknown thinking level '{escape(level)}'.[/bold red] "
                    f"Choose one of: {', '.join(levels) or '(this model accepts none)'}"
                )
            return self._prompt_and_handle_slash_commands(prompt)
        if user_input == "/model" or user_input.startswith("/model "):
            self._handle_model_command(user_input[6:].strip())
            return self._prompt_and_handle_slash_commands(prompt)
        if user_input in self._MODE_COMMANDS_MAPPING:
            if self.config.mode == self._MODE_COMMANDS_MAPPING[user_input]:
                return self._prompt_and_handle_slash_commands(
                    f"[bold red]Already in {self.config.mode} mode.[/bold red]\n{prompt}"
                )
            self.config.mode = self._MODE_COMMANDS_MAPPING[user_input]
            console.print(f"Switched to [bold green]{self.config.mode}[/bold green] mode.")
            return user_input
        return user_input