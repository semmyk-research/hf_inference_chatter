"""Validate provider names against Hugging Face Inference Providers list.

Source: https://huggingface.co/docs/inference-providers/index

Functions:
- get_supported_providers() -> set[str]
- normalize_provider(text: str) -> str | None
- is_valid_provider(text: str) -> bool
- suggest_providers(text: str, limit: int = 3) -> list[str]

Supports common aliases (e.g., "together-ai" -> "together", "fireworks" -> "fireworks-ai").
"""

from __future__ import annotations

from difflib import get_close_matches
from typing import Iterable


# Canonical provider slugs from docs (table and provider URLs)
_CANONICAL: set[str] = {
    "cerebras",
    "cohere",
    "fal-ai",
    "featherless-ai",
    "fireworks-ai",
    "groq",
    "hf-inference",
    "hyperbolic",
    "nebius",
    "novita",
    "nscale",
    "replicate",
    "sambanova",
    "together",
}

# Common aliases users may type; maps to canonical slug
_ALIASES: dict[str, str] = {
    "together-ai": "together",
    "fireworks": "fireworks-ai",
    "falai": "fal-ai",
    "featherless": "featherless-ai",
    "hf": "hf-inference",
    "huggingface": "hf-inference",
}


def _to_key(text: str) -> str:
    return (text or "").strip().lower()


def get_supported_providers(extra: Iterable[str] | None = None) -> set[str]:
    """Return set of canonical provider slugs.

    Optionally extend with additional slugs via `extra`.
    """
    return _CANONICAL | set(map(_to_key, (extra or [])))


def normalize_provider(text: str) -> str | None:
    """Return canonical provider slug for `text`, if known; else None.

    Accepts canonical slugs and common aliases.
    """
    key = _to_key(text)
    if not key:
        return None
    if key in _CANONICAL:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    return None


def is_valid_provider(text: str) -> bool:
    """True if `text` is a known provider or alias."""
    return normalize_provider(text) is not None


def suggest_providers(text: str, limit: int = 3) -> list[str]:
    """Suggest close canonical matches for `text`.

    Uses difflib to match against canonical slugs; returns up to `limit` suggestions.
    """
    key = _to_key(text)
    if not key:
        return []
    # Search both canonical and alias keys to be helpful, then map to canonical
    candidates = list(_CANONICAL | set(_ALIASES))
    suggestions = get_close_matches(key, candidates, n=limit, cutoff=0.6)
    canon = []
    for s in suggestions:
        canon_slug = s if s in _CANONICAL else _ALIASES.get(s)
        if canon_slug and canon_slug not in canon:
            canon.append(canon_slug)
    return canon[:limit]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:])
    if not query:
        print("Usage: python provider_validator.py <provider-name>")
        raise SystemExit(2)

    norm = normalize_provider(query)
    if norm:
        print(f"valid: {norm}")
    else:
        print("invalid")
        suggestions = suggest_providers(query)
        if suggestions:
            print("did_you_mean:", ", ".join(suggestions))

