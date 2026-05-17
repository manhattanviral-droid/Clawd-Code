"""
MCP server loader for Clawd Code.

Reads `.mcp.json` from the current project (Claude Code-compatible schema),
spawns each declared server as a stdio subprocess, and speaks the MCP
JSON-RPC 2.0 protocol over its stdin/stdout. Resulting clients are placed
into ToolContext.mcp_clients so the existing `MCP` tool can dispatch calls.

This was added by the Manhattan Viral fork — upstream GPT-AGI/Clawd-Code
ships the MCP *tool dispatcher* but not the *server connection layer*.

Schema (matches Claude Code's project-scoped .mcp.json):

    {
      "mcpServers": {
        "make": {
          "command": "npx",
          "args": ["-y", "@makehq/mcp-server@latest"],
          "env": { "MAKE_API_TOKEN": "${MAKE_API_TOKEN}" }
        }
      }
    }

Env-var interpolation: ${VAR} and ${VAR:-default} are expanded from the
parent process environment before being passed to the child.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


_ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}", re.IGNORECASE)


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:-default} in strings/dicts/lists."""
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            var, default = m.group(1), m.group(2) or ""
            return os.environ.get(var, default)
        return _ENV_VAR_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


class StdioMCPClient:
    """
    Minimal MCP client that talks JSON-RPC 2.0 over a child process's stdio.
    Implements the surface required by ToolContext.mcp_clients consumers:
      - list_tools() -> list[str]
      - call_tool(name: str, args: dict) -> Any
    """

    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str]):
        self.name = name
        merged_env = {**os.environ, **{k: str(v) for k, v in env.items() if v is not None}}
        self.proc = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
            text=True,
            bufsize=1,
        )
        self._id = 0
        self._lock = threading.Lock()
        self._tools: list[str] | None = None
        self._initialize()

    def _next_id(self) -> int:
        with self._lock:
            self._id += 1
            return self._id

    def _send(self, method: str, params: dict | None = None, is_notification: bool = False) -> dict | None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        if not is_notification:
            payload["id"] = self._next_id()
        line = json.dumps(payload) + "\n"
        assert self.proc.stdin is not None
        self.proc.stdin.write(line)
        self.proc.stdin.flush()
        if is_notification:
            return None
        return self._read_response(payload["id"])

    def _read_response(self, expected_id: int, timeout: float = 30.0) -> dict:
        assert self.proc.stdout is not None
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"MCP server '{self.name}' exited prematurely")
                time.sleep(0.05)
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # ignore non-JSON noise (e.g. log lines)
            if msg.get("id") == expected_id:
                if "error" in msg:
                    raise RuntimeError(f"MCP error from {self.name}: {msg['error']}")
                return msg.get("result", {})
        raise TimeoutError(f"MCP server '{self.name}' did not respond within {timeout}s")

    def _initialize(self) -> None:
        self._send("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "clawd-py", "version": "1.0.0"},
        })
        self._send("notifications/initialized", is_notification=True)

    def list_tools(self) -> list[str]:
        if self._tools is None:
            result = self._send("tools/list") or {}
            tools = result.get("tools", [])
            self._tools = [t.get("name", "") for t in tools if t.get("name")]
        return list(self._tools)

    def call_tool(self, tool_name: str, args: dict) -> Any:
        result = self._send("tools/call", {"name": tool_name, "arguments": args}) or {}
        return result.get("content", result)

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def find_mcp_config(start: Path | None = None) -> Path | None:
    """
    Walk up from `start` looking for .mcp.json. Falls back to
    ~/.clawd/mcp.json for a user-level default.
    """
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        candidate = parent / ".mcp.json"
        if candidate.is_file():
            return candidate
    user_default = Path.home() / ".clawd" / "mcp.json"
    if user_default.is_file():
        return user_default
    return None


def load_mcp_clients(start: Path | None = None) -> dict[str, StdioMCPClient]:
    """
    Locate .mcp.json, spawn all declared servers, return {name: client}.
    Servers that fail to start are skipped with a printed warning so a bad
    server config doesn't take down the REPL.
    """
    cfg_path = find_mcp_config(start)
    if cfg_path is None:
        return {}
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception as exc:
        print(f"[clawd-py] Failed to parse {cfg_path}: {exc}")
        return {}

    servers = cfg.get("mcpServers") or {}
    if not isinstance(servers, dict):
        print(f"[clawd-py] {cfg_path}: 'mcpServers' must be an object")
        return {}

    clients: dict[str, StdioMCPClient] = {}
    for name, spec in servers.items():
        if not isinstance(spec, dict):
            continue
        command = _expand_env(spec.get("command"))
        args = _expand_env(spec.get("args") or [])
        env = _expand_env(spec.get("env") or {})
        if not command:
            print(f"[clawd-py] MCP server '{name}' missing 'command' — skipping")
            continue
        try:
            clients[name] = StdioMCPClient(name, command, list(args), dict(env))
            tool_count = len(clients[name].list_tools())
            print(f"[clawd-py] MCP connected: {name} ({tool_count} tools)")
        except Exception as exc:
            print(f"[clawd-py] MCP server '{name}' failed to start: {exc}")
    return clients
