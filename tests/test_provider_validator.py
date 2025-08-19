from __future__ import annotations

from provider_validator import (
    get_supported_providers,
    normalize_provider,
    is_valid_provider,
    suggest_providers,
)


def test_normalize_and_validate():
    assert normalize_provider("fireworks-ai") == "fireworks-ai"
    assert normalize_provider("fireworks") == "fireworks-ai"
    assert is_valid_provider("together-ai") is True
    assert is_valid_provider("together") is True
    assert is_valid_provider("unknown-provider") is False


def test_suggestions():
    sugg = suggest_providers("replicat")
    assert "replicate" in sugg


def test_supported_extend():
    base = get_supported_providers()
    extended = get_supported_providers(["my-provider"])
    assert "my-provider" not in base
    assert "my-provider" in extended

