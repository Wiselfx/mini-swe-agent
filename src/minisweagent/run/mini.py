#!/usr/bin/env python3

"""Run mini-SWE-agent in your local environment. This is the default executable `mini`."""
# Read this first: https://mini-swe-agent.com/latest/usage/mini/  (usage)

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from minisweagent.agents import get_agent
from minisweagent.agents.utils.prompt_user import _multiline_prompt
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.models.utils.content_string import get_content_string
from minisweagent.run.utilities.config import configure_if_first_time
from minisweagent.run.utilities.sessions import cwd_dir, new_session_path, pick_session
from minisweagent.utils.serialize import UNSET, recursive_merge

DEFAULT_CONFIG_FILE = Path(os.getenv("MSWEA_MINI_CONFIG_PATH", builtin_config_dir / "mini.yaml"))


_HELP_TEXT = """Run mini-SWE-agent in your local environment.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/mini/[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c mini.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c mini.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml agent.mode=yolo[/bold green]
"""

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use",),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'litellm' or 'minisweagent.models.litellm_model.LitellmModel')", rich_help_panel="Advanced"),
    agent_class: str | None = typer.Option(None, "--agent-class", help="Agent class to use (e.g., 'interactive' or 'minisweagent.agents.interactive.InteractiveAgent')", rich_help_panel="Advanced"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment class to use (e.g., 'local' or 'minisweagent.environments.local.LocalEnvironment')", rich_help_panel="Advanced"),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output trajectory file (default: a new session file). With -r: the session to resume."),
    exit_immediately: bool = typer.Option(False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Advanced"),
    resume: Annotated[bool, typer.Option("-r", "--resume", help="Resume a session: pick one, or pass its file with -o.")] = False,
) -> Any:
    # fmt: on
    configure_if_first_time()

    # Build the config from the command line arguments
    console.print(f"Building agent config from specs: [bold green]{config_spec}[/bold green]")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    saved = None
    if resume:
        if output is None and not sys.stdin.isatty():
            raise typer.BadParameter("Without a terminal, pass the session file to resume with -o.")
        output = output or pick_session()
        saved = json.loads(output.read_text())
        # Messages carry provider-specific fields, so resume with the model setup the session ran with
        saved_model = saved.get("info", {}).get("config", {})
        configs.append({"model": {**saved_model.get("model", {}), "model_class": saved_model.get("model_type", UNSET)}})
        console.print(f"Resuming [bold green]{output}[/bold green]")
        if output.parent.parent == cwd_dir().parent and output.parent != cwd_dir():
            console.print(f"[bold yellow]This session ran in another directory ({output.parent.name}).[/bold yellow]")
    configs.append({
        "run": {
            "task": task or UNSET,
        },
        "agent": {
            "agent_class": agent_class or UNSET,
            "mode": "yolo" if yolo else UNSET,
            "cost_limit": cost_limit if cost_limit is not None else UNSET,
            "confirm_exit": False if exit_immediately or not sys.stdin.isatty() else UNSET,  # no one to ask
            "output_path": output or UNSET,
        },
        "model": {
            "model_class": model_class or UNSET,
            "model_name": model_name or UNSET,
        },
        "environment": {
            "environment_class": environment_class or UNSET,
        },
    })
    config = recursive_merge(*configs)

    if (run_task := config.get("run", {}).get("task", UNSET)) is UNSET:
        run_task = UNSET  # prompted for below, after model/effort selection

    model = get_model(config=config.get("model", {}))
    env = get_environment(config.get("environment", {}), default_type="local")
    agent = get_agent(model, env, config.get("agent", {}), default_type="interactive")

    if saved:
        agent.cost = saved["info"]["model_stats"]["instance_cost"]
        agent.n_calls = saved["info"]["model_stats"]["api_calls"]
        agent.compacted_history = saved.get("compacted_history", [])
        if last := next((m for m in reversed(saved["messages"]) if m.get("role") == "assistant" and get_content_string(m).strip()), None):
            console.print("[bold]Last reply:[/bold]")
            console.print(get_content_string(last).strip()[-2000:], markup=False)

    if run_task is UNSET:
        if getattr(agent, "preflight", None) and sys.stdin.isatty() and not saved:
            agent.preflight()
        console.print("[bold yellow]What do you want to do?[/bold yellow]")
        run_task = _multiline_prompt()
        console.print("[bold green]Got that, thanks![/bold green]")

    if agent.config.output_path is None:
        agent.config.output_path = new_session_path(run_task)
    agent.run(run_task, messages=saved["messages"] if saved else None)
    if (output_path := agent.config.output_path):
        console.print(f"Saved trajectory to [bold green]'{output_path}'[/bold green]")
    return agent


if __name__ == "__main__":
    app()
