"""
Compatibility shim — the canonical config now lives at the repo root in
`config.py`. This module re-exports the same `Settings` class and `settings`
instance so existing imports (`from chatbot.config import settings`) keep
working unchanged.
"""

from __future__ import annotations

from config import Settings, settings  # noqa: F401

__all__ = ["Settings", "settings"]
