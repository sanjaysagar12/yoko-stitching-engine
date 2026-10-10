"""Tool registry: ``@register("endLine")`` on a Tool subclass; unknown type -> UnsupportedToolError."""

from __future__ import annotations

from collections.abc import Callable

from ..errors import UnsupportedToolError
from .base import Tool


class Registry:
    """Maps a sm2d ``type`` attribute to a Tool instance."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool_type: str) -> Callable[[type[Tool]], type[Tool]]:
        """Class decorator registering a Tool subclass under ``tool_type``."""

        def deco(cls: type[Tool]) -> type[Tool]:
            cls.type = tool_type
            self._tools[tool_type] = cls()
            return cls

        return deco

    def get(self, tool_type: str) -> Tool:
        """Tool for ``tool_type`` or UnsupportedToolError."""
        try:
            return self._tools[tool_type]
        except KeyError:
            raise UnsupportedToolError(tool_type or "<none>") from None

    def types(self) -> list[str]:
        """Registered tool types."""
        return sorted(self._tools)


default_registry = Registry()
register = default_registry.register
