"""Helper client for interacting with the nu Knowledge Base MCP server.

This module provides a minimal wrapper that spawns the MCP stdio server
exposed by `nu-knowledge-base-mcp-nu-cli` and calls the registered tools
programmatically from the Slack bot.
"""

from __future__ import annotations

import json
import os
import shlex
from typing import Any, Dict, Optional

import anyio
from mcp import ClientSession, StdioServerParameters, stdio_client

JsonDict = Dict[str, Any]


class KnowledgeBaseMCPClient:
    """Thin wrapper around the nu Knowledge Base MCP stdio server."""

    def __init__(
        self,
        *,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        default_kb: Optional[str] = None,
    ) -> None:
        if not command:
            raise ValueError("command is required for KnowledgeBaseMCPClient")

        self.command = command
        self.args = list(args or [])
        self.env = {
            key: value
            for key, value in (env or os.environ).items()
            if value is not None
        }
        self.default_kb = default_kb

    def search(
        self,
        query: str,
        *,
        kb_name: Optional[str] = None,
        index_name: Optional[str] = None,
        num_results: int = 5,
        expand_window: Optional[int] = None,
        filters: Optional[JsonDict] = None,
        logger=None,
    ) -> JsonDict:
        """Call the `searchKnowledgeBase` MCP tool and return the JSON payload."""

        effective_kb = kb_name or self.default_kb
        if not effective_kb:
            raise ValueError("kb_name is required when using the MCP client")

        arguments: JsonDict = {
            "kb_name": effective_kb,
            "query": query,
            "num_results": num_results,
        }

        if index_name:
            arguments["index_name"] = index_name
        if expand_window is not None:
            arguments["expand_window"] = expand_window
        if filters:
            arguments["filter"] = filters

        try:
            return anyio.run(self._search, arguments, logger)
        except RuntimeError as exc:  # pragma: no cover - defensive fallback
            # anyio.run raises RuntimeError if called from an existing running loop
            if "event loop" in str(exc).lower():
                raise RuntimeError(
                    "KnowledgeBaseMCPClient.search cannot be called while an event loop is running"
                ) from exc
            raise

    def list_knowledge_bases(self, logger=None) -> JsonDict:
        """Call the `listKnowledgeBases` MCP tool."""

        return anyio.run(self._list_kbs, logger)

    def _open_session(self):
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )

        return stdio_client(server_params)

    async def _search(self, arguments: JsonDict, logger) -> JsonDict:
        async with self._open_session() as (read_stream, write_stream):
            session = ClientSession(read_stream, write_stream)
            init_result = await session.initialize()

            if logger:
                server_info = getattr(init_result, "serverInfo", None)
                server_name = getattr(server_info, "name", "unknown")
                server_version = getattr(server_info, "version", "")
                logger.debug(
                    "Connected to KB MCP server", extra={"server": server_name, "version": server_version}
                )

            call_result = await session.call_tool("searchKnowledgeBase", arguments)
            if call_result.isError:
                raise RuntimeError("MCP searchKnowledgeBase responded with isError=true")

            return self._decode_call_result(call_result, logger)

    async def _list_kbs(self, logger) -> JsonDict:
        async with self._open_session() as (read_stream, write_stream):
            session = ClientSession(read_stream, write_stream)
            await session.initialize()
            call_result = await session.call_tool("listKnowledgeBases")
            if call_result.isError:
                raise RuntimeError("MCP listKnowledgeBases responded with isError=true")
            return self._decode_call_result(call_result, logger)

    @staticmethod
    def _decode_call_result(call_result, logger) -> JsonDict:
        text_parts: list[str] = []

        for item in call_result.content:
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text_parts.append(item.text)
            elif item_type == "image":
                if logger:
                    logger.warning("Ignoring unexpected image content from MCP tool")
            elif item_type == "resource":
                if logger:
                    logger.warning("Ignoring unexpected resource content from MCP tool")

        if not text_parts:
            return {}

        raw_text = "".join(text_parts).strip()
        if not raw_text:
            return {}

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            if logger:
                logger.error("Unable to decode MCP response as JSON", extra={"payload": raw_text})
            raise RuntimeError("Invalid JSON returned by MCP tool") from exc

        return {"raw": data, "results": data.get("results", [])}


def parse_server_args(raw_args: str | None) -> list[str]:
    """Split a shell-style argument string into a list."""

    if not raw_args:
        return []

    return shlex.split(raw_args)

