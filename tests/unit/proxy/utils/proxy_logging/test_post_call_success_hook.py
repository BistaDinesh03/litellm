"""Pin ``ProxyLogging.post_call_success_hook``."""

from __future__ import annotations

from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import CallTypes, UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs, ModelResponse


class _CountingModerationGuardrail(CustomGuardrail):
    """Like openai_moderation: only apply_guardrail, counting every provider call."""

    def __init__(self, *, parallel: bool, block: bool = False) -> None:
        super().__init__(guardrail_name="post-mod", event_hook=GuardrailEventHooks.post_call, default_on=False)
        self.evaluations = 0
        self.block = block
        self.run_in_parallel = parallel

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: str,
        logging_obj: object = None,
    ) -> GenericGuardrailAPIInputs:
        self.evaluations += 1
        if self.block:
            raise HTTPException(status_code=400, detail={"error": "Violated moderation policy"})
        return inputs


def _proxy_request(metadata_key: str, *, request_level: bool) -> dict[str, Any]:
    return {
        "model": "m",
        "proxy_server_request": {"url": "http://localhost:4000/v1/chat/completions"},
        metadata_key: {"guardrails": ["post-mod"]},
        **({"guardrails": ["post-mod"]} if request_level else {}),
    }


async def _run_both_layers(
    proxy_logging: ProxyLogging, guardrail: _CountingModerationGuardrail, data: dict[str, Any], response: ModelResponse
) -> object:
    """The router spreads the deployment's guardrails into kwargs before litellm.acompletion
    fires the deployment hook; the proxy hook then sees the final response."""
    await guardrail.async_post_call_success_deployment_hook(
        request_data={**data, "guardrails": ["post-mod"]}, response=response, call_type=CallTypes.acompletion
    )
    return await proxy_logging.post_call_success_hook(data=data, response=response, user_api_key_dict=UserAPIKeyAuth())


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("request_level", [False, True], ids=["model-level", "model-and-request-level"])
async def test_model_level_post_call_guardrail_runs_once_per_proxy_request(
    proxy_logging: ProxyLogging, monkeypatch: pytest.MonkeyPatch, parallel: bool, metadata_key: str, request_level: bool
) -> None:
    """LIT-10481: one openai_moderation post_call guardrail attached through litellm_params.guardrails
    called the moderation provider twice per non-streaming chat request after #40154."""
    guardrail: Final = _CountingModerationGuardrail(parallel=parallel)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    response: Final = ModelResponse(choices=[{"message": {"role": "assistant", "content": "Hello"}}])

    out: Final = await _run_both_layers(
        proxy_logging, guardrail, _proxy_request(metadata_key, request_level=request_level), response
    )

    assert out is response
    assert guardrail.evaluations == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
async def test_blocking_model_level_post_call_guardrail_still_blocks_once(
    proxy_logging: ProxyLogging, monkeypatch: pytest.MonkeyPatch, parallel: bool
) -> None:
    guardrail: Final = _CountingModerationGuardrail(parallel=parallel, block=True)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    response: Final = ModelResponse(choices=[{"message": {"role": "assistant", "content": "Hello"}}])

    with pytest.raises(HTTPException) as exc_info:
        await _run_both_layers(proxy_logging, guardrail, _proxy_request("metadata", request_level=False), response)

    assert exc_info.value.status_code == 400
    assert guardrail.evaluations == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
async def test_proxy_rechecks_a_response_rewritten_after_the_deployment_checked_it(
    proxy_logging: ProxyLogging, monkeypatch: pytest.MonkeyPatch, parallel: bool
) -> None:
    """The deployment's check covers only the content it saw. A later hook that rewrites the
    response (for example, unmasking PII) must not let the proxy skip the new content."""
    guardrail: Final = _CountingModerationGuardrail(parallel=parallel)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    response: Final = ModelResponse(choices=[{"message": {"role": "assistant", "content": "Hello"}}])
    data: Final = _proxy_request("metadata", request_level=False)

    await guardrail.async_post_call_success_deployment_hook(
        request_data={**data, "guardrails": ["post-mod"]}, response=response, call_type=CallTypes.acompletion
    )
    rewritten: Final = ModelResponse(choices=[{"message": {"role": "assistant", "content": "Hello, rewritten"}}])
    await proxy_logging.post_call_success_hook(data=data, response=rewritten, user_api_key_dict=UserAPIKeyAuth())

    assert guardrail.evaluations == 2


@pytest.mark.asyncio
async def test_sdk_request_without_proxy_is_checked_at_the_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct litellm.acompletion callers have no proxy hook, so the deployment hook must still run."""
    guardrail: Final = _CountingModerationGuardrail(parallel=False)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    response: Final = ModelResponse(choices=[{"message": {"role": "assistant", "content": "Hello"}}])

    await guardrail.async_post_call_success_deployment_hook(
        request_data={"model": "m", "guardrails": ["post-mod"], "metadata": {}},
        response=response,
        call_type=CallTypes.acompletion,
    )

    assert guardrail.evaluations == 1


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    ProxyLogging._callback_capabilities_cache.clear()
    yield
    ProxyLogging._callback_capabilities_cache.clear()


def _make_guardrail(name="g", should_run=True, override=None):
    cb = MagicMock(spec=CustomGuardrail)
    cb.__class__ = CustomGuardrail
    cb.guardrail_name = name
    cb.event_hook = GuardrailEventHooks.post_call
    cb.should_run_guardrail = MagicMock(return_value=should_run)
    cb.async_post_call_success_hook = AsyncMock(return_value=override)
    cb.post_call_already_checked = MagicMock(return_value=False)
    cb.run_in_parallel = False
    return cb


@pytest.mark.asyncio
async def test_post_call_success_hook_returns_response_when_no_callbacks(proxy_logging, make_user_api_key_auth, mock_callbacks_disabled):
    response = {"original": True, "model": "m", "choices": []}
    out = await proxy_logging.post_call_success_hook(
        data={}, response=response, user_api_key_dict=make_user_api_key_auth()
    )
    assert out == {"original": True, "model": "m", "choices": []}


@pytest.mark.asyncio
async def test_post_call_success_hook_runs_other_callback_and_replaces_response(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    new_response = {"modified": True, "kept": "yes", "final": "v"}

    class _CL(CustomLogger):
        async def async_post_call_success_hook(self, **kwargs):  # type: ignore[override]
            return new_response

    monkeypatch.setattr(litellm, "callbacks", [_CL()])
    out = await proxy_logging.post_call_success_hook(
        data={}, response={"original": True}, user_api_key_dict=make_user_api_key_auth()
    )
    assert out == new_response


@pytest.mark.asyncio
async def test_post_call_success_hook_guardrail_should_not_run_skipped(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    g = _make_guardrail(should_run=False)
    monkeypatch.setattr(litellm, "callbacks", [g])
    response = MagicMock()
    out = await proxy_logging.post_call_success_hook(
        data={}, response=response, user_api_key_dict=make_user_api_key_auth()
    )
    g.async_post_call_success_hook.assert_not_called()
    assert out is response


@pytest.mark.asyncio
async def test_post_call_success_hook_guardrail_error_raises(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    g = _make_guardrail()
    g.async_post_call_success_hook = AsyncMock(side_effect=RuntimeError("blocked"))
    monkeypatch.setattr(litellm, "callbacks", [g])
    with pytest.raises(RuntimeError):
        await proxy_logging.post_call_success_hook(
            data={}, response=MagicMock(), user_api_key_dict=make_user_api_key_auth()
        )


@pytest.mark.asyncio
async def test_post_call_success_hook_guardrail_returns_modified_response(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    modified = {"a": 1, "b": 2, "c": 3}
    g = _make_guardrail(override=modified)
    monkeypatch.setattr(litellm, "callbacks", [g])
    out = await proxy_logging.post_call_success_hook(
        data={}, response={"orig": True}, user_api_key_dict=make_user_api_key_auth()
    )
    assert out == modified


@pytest.mark.asyncio
@pytest.mark.parametrize("run_in_parallel", [False, True], ids=["sequential", "parallel"])
async def test_post_call_block_names_the_blocking_guardrail_in_applied_guardrails(
    proxy_logging, make_user_api_key_auth, monkeypatch, run_in_parallel
):
    def _passer_that_records(data, user_api_key_dict, response):
        data["metadata"]["applied_guardrails"] = ["passer"]

    passer = _make_guardrail("passer")
    passer.async_post_call_success_hook = AsyncMock(side_effect=_passer_that_records)
    passer.run_in_parallel = run_in_parallel
    blocker = _make_guardrail("blocker")
    blocker.async_post_call_success_hook = AsyncMock(side_effect=HTTPException(status_code=400, detail="blocked"))
    blocker.run_in_parallel = run_in_parallel
    monkeypatch.setattr(litellm, "callbacks", [passer, blocker])
    data = {"model": "m", "metadata": {}}
    with pytest.raises(HTTPException):
        await proxy_logging.post_call_success_hook(
            data=data, response=MagicMock(), user_api_key_dict=make_user_api_key_auth()
        )
    assert data["metadata"]["applied_guardrails"] == ["passer", "blocker"]
