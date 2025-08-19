from __future__ import annotations

import types
import builtins
import pytest

import os
from hf_client import HFChatClient


def test_normalize_history_tuples():
    client = HFChatClient()
    messages = client._normalize_history(
        history=[["hi", "hello"], ["how are you?", None]],
        system_message="be nice",
        latest_user_message="tell me a joke",
    )
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[-1]["content"] == "tell me a joke"


def test_select_target_model():
    assert HFChatClient._select_target(
        backend="model", model_id="m", provider=None, endpoint_url=None, default_model="d"
    ) == "m"
    assert HFChatClient._select_target(
        backend="model", model_id=None, provider=None, endpoint_url=None, default_model="d"
    ) == "d"


def test_select_target_provider():
    assert HFChatClient._select_target(
        backend="provider", model_id="m", provider="p", endpoint_url=None, default_model="d"
    ) == "m:p"
    with pytest.raises(ValueError):
        HFChatClient._select_target("provider", model_id=None, provider="p", endpoint_url=None, default_model="d")


def test_select_target_endpoint():
    assert HFChatClient._select_target(
        backend="endpoint", model_id=None, provider=None, endpoint_url="http://x", default_model="d"
    ) == "http://x"
    with pytest.raises(ValueError):
        HFChatClient._select_target("endpoint", model_id=None, provider=None, endpoint_url=None, default_model="d")


def test_logout_clears_token_env(monkeypatch):
    # Stub network-affecting calls
    monkeypatch.setattr("hf_client.login", lambda *_, **__: None)
    monkeypatch.setattr("hf_client.hf_logout", lambda *_, **__: None)

    client = HFChatClient(token="fake")
    # Ensure env vars are present
    monkeypatch.setenv("HF_TOKEN", "fake")
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "fake")
    ok = client.logout()
    assert ok is True
    assert "HF_TOKEN" not in os.environ
    assert "HUGGINGFACEHUB_API_TOKEN" not in os.environ


