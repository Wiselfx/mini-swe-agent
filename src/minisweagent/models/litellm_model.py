import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import litellm
from pydantic import BaseModel

from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_model")

if os.getenv("MSWEA_QUIET", "").strip().lower() not in {"", "0", "false", "no", "off"}:
    # Quiet mode: litellm prints a "Provider List: ..." block whenever it cannot resolve
    # cost metadata for an unregistered model (local servers, most often).
    litellm.suppress_debug_info = True
    # ...and a warning per call for every param its provider definition would drop,
    # which is exactly what `drop_params: true` is for. Not worth a line per step.
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)


class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name. Highly recommended to include the provider in the model name, e.g., `anthropic/claude-sonnet-4-5-20250929`."""
    model_kwargs: dict[str, Any] = {}
    """Additional arguments passed to the API."""
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    """Model registry for cost tracking and model metadata. See the local model guide (https://mini-swe-agent.com/latest/models/local_models/) for more details."""
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""
    context_window: int | None = None
    """Context window in tokens, for the status footer. Looked up from litellm's registry when unset."""
    pricing: dict[str, float] | None = None
    """Price overrides in USD per million tokens: `input`, `output`, `cache_read`, `cache_write`.

    Use when litellm's registry has no entry for the model, or when it disagrees with the bill.
    """
    model_profiles: dict[str, dict[str, Any]] = {}
    """Named alternate models, switchable at runtime with `/model` in interactive mode.

    Each value is a dict of model config fields to override, e.g.
    `{"beam-free": {"model_name": "opencode_go/beam-free", "cost_tracking": "ignore_errors"}}`.
    """
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class LitellmModel:
    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self._context_window_cache: int | None = None
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def get_context_window(self) -> int | None:
        """Max input tokens for the current model, from config or litellm's registry."""
        if self.config.context_window:
            return self.config.context_window
        if self._context_window_cache is None:
            try:
                info = litellm.get_model_info(self.config.model_name)
            except Exception:
                info = {}
            self._context_window_cache = int(info.get("max_input_tokens") or 0)
        return self._context_window_cache or None

    @staticmethod
    def _extract_usage(response) -> dict[str, int]:
        """Normalize token accounting across OpenAI-style and Anthropic-style responses."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}

        def get(*names: str) -> int:
            for name in names:
                value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return int(value)
            return 0

        try:
            details = getattr(usage, "prompt_tokens_details", None) or getattr(usage, "input_tokens_details", None)
            cached = get("cache_read_input_tokens")
            if not cached and details is not None:
                value = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    cached = int(value)
            return {
                "prompt_tokens": get("prompt_tokens", "input_tokens"),
                "completion_tokens": get("completion_tokens", "output_tokens"),
                "cached_tokens": cached,
                "cache_write_tokens": get("cache_creation_input_tokens"),
            }
        except (TypeError, ValueError):
            return {}

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[BASH_TOOL],
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        # Note: all model.query() implementations must persist the response and cost on FormatError.
        try:
            actions = self._parse_actions(response)
        except FormatError as e:
            e.messages[0]["extra"].update(cost_output)
            e.messages[0]["extra"]["usage"] = self._extract_usage(response)
            try:
                e.messages[0]["extra"]["response"] = response.model_dump(mode="json")
            except Exception:
                # model_dump failed (e.g. unserializable object); fall back to repr
                # so the spec contract ("response MUST be persisted") holds unconditionally.
                e.messages[0]["extra"]["response"] = repr(response)
            raise
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": actions,
            "response": response.model_dump(),
            "usage": self._extract_usage(response),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def apply_profile(self, name: str) -> str:
        """Switch this model to a named entry from `config.model_profiles` in place. Returns its model name."""
        if name not in self.config.model_profiles:
            raise KeyError(name)
        for field, value in self.config.model_profiles[name].items():
            if field not in type(self.config).model_fields:
                raise ValueError(f"Unknown field {field!r} in model profile {name!r}")
            setattr(self.config, field, value)
        return self.config.model_name

    def _per_token_prices(self) -> dict[str, float] | None:
        """USD per token for input/output/cache_read/cache_write, from config overrides or litellm's registry."""
        if self.config.pricing:
            return {key: float(value) / 1e6 for key, value in self.config.pricing.items()}
        try:
            info = litellm.get_model_info(self.config.model_name)
        except Exception:
            return None
        if info.get("input_cost_per_token") is None or info.get("output_cost_per_token") is None:
            return None
        return {
            "input": info["input_cost_per_token"],
            "output": info["output_cost_per_token"],
            "cache_read": info.get("cache_read_input_token_cost") or 0.0,
            "cache_write": info.get("cache_creation_input_token_cost") or 0.0,
        }

    def _cost_from_usage(self, response) -> float | None:
        """Cost including cache-read discounts. Returns None when the usage or prices are unusable."""
        usage = self._extract_usage(response)
        prices = self._per_token_prices()
        prompt, completion = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        if prices is None or not prompt:
            return None
        cached = usage.get("cached_tokens", 0)
        return (
            (prompt - cached) * prices["input"]
            + cached * prices.get("cache_read", 0.0)
            + completion * prices["output"]
            + usage.get("cache_write_tokens", 0) * prices.get("cache_write", 0.0)
        )

    def _calculate_cost(self, response) -> dict[str, float]:
        try:
            # litellm's calculator bills every prompt token at the full input price, which overstates
            # cost badly on providers with a cheap cache-read rate (or with a wrong catalog entry).
            cost = self._cost_from_usage(response)
            if cost is None:
                cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        return {"cost": cost}

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(
            tool_calls,
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": response.choices[0].finish_reason},
        )

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
