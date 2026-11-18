"""Cost accounting: cache-read discounts and per-model price overrides."""

from types import SimpleNamespace

import pytest

from minisweagent.models.litellm_model import LitellmModel

GO_KWARGS = {"litellm_session_id": "test", "drop_params": True}


def make_model(name: str, **kwargs) -> LitellmModel:
    return LitellmModel(model_name=name, model_kwargs=GO_KWARGS, cost_tracking="ignore_errors", **kwargs)


def make_response(prompt: int, completion: int, cached: int = 0):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        )
    )


def test_cost_does_not_bill_cached_tokens_at_full_input_price():
    model = make_model("opencode_go/glm-5.3")
    prices = model._per_token_prices()
    response = make_response(prompt=1000, completion=100, cached=900)
    cost = model._calculate_cost(response)["cost"]
    full_price = 1000 * prices["input"] + 100 * prices["output"]
    expected = 100 * prices["input"] + 900 * prices["cache_read"] + 100 * prices["output"]
    assert cost == pytest.approx(expected)
    assert cost < full_price / 2


def test_cost_matches_full_price_without_cache_hits():
    model = make_model("opencode_go/glm-5.3")
    prices = model._per_token_prices()
    cost = model._calculate_cost(make_response(prompt=1000, completion=100, cached=0))["cost"]
    assert cost == pytest.approx(1000 * prices["input"] + 100 * prices["output"])


def test_pricing_override_beats_registry():
    """litellm's registry lists deepseek-v4.1-flash at twice the price on this plan."""
    response = make_response(prompt=1006, completion=161, cached=768)
    catalog = make_model("opencode_go/deepseek-v4.1-flash")._calculate_cost(response)["cost"]
    plan = make_model(
        "opencode_go/deepseek-v4.1-flash", pricing={"input": 0.15, "output": 0.6, "cache_read": 0.003}
    )._calculate_cost(response)["cost"]
    assert plan == pytest.approx(238 * 0.15e-6 + 768 * 0.003e-6 + 161 * 0.6e-6)
    # the plan price is exactly half of litellm's catalog entry for this model
    assert plan == pytest.approx(catalog / 2)


def test_free_model_reports_zero_cost_without_raising():
    assert make_model("opencode_go/beam-free")._calculate_cost(make_response(500, 50, 100))["cost"] == 0.0
