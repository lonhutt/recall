#!/usr/bin/env python3
"""Call a Recall MCP tool directly over HTTP, without an active MCP session.

Why this exists: Recall's tools are normally reached through Claude Code's
own MCP connection, but that connection only exists in sessions where the
`recall` MCP server was registered *before* the session started. This script
does the same handshake (POST /mcp -> capture Mcp-Session-Id -> POST
tools/call with that header) so you can still save/search/list/etc. from a
shell, a script, or a session where the MCP connection isn't up.

Usage:
  recall_cli.py <tool> [json-args]

Examples:
  recall_cli.py list_memories '{"limit": 5}'
  recall_cli.py search_memory '{"query": "how does Lon like commits structured"}'
  recall_cli.py save_memory '{"type": "project", "slug": "x", "description": "d", "body": "b"}'
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8092/mcp"
DEFAULT_ENV_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".env")


def read_env_file(path):
    values = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k] = v
    return values


def post(url, token, session_id, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json, text/event-stream")
    if session_id:
        req.add_header("Mcp-Session-Id", session_id)
    try:
        with urllib.request.urlopen(req) as resp:
            new_session_id = resp.headers.get("Mcp-Session-Id", session_id)
            body = resp.read().decode()
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("error: 401 unauthorized - the token doesn't match RECALL_HTTP_TOKEN in .env "
                  "(check it hasn't been rotated)", file=sys.stderr)
        else:
            print(f"error: HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"error: could not reach {url}: {e.reason}\n"
              f"is the stack up? try: docker compose ps", file=sys.stderr)
        sys.exit(1)

    for line in body.splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: "):]), new_session_id
    raise RuntimeError(f"no SSE data line in response: {body!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tool", help="Tool name, e.g. search_memory, save_memory, list_memories")
    parser.add_argument("args_json", nargs="?", default="{}", help="JSON object of tool arguments")
    parser.add_argument("--url", default=os.environ.get("RECALL_URL", DEFAULT_URL))
    parser.add_argument("--token", default=os.environ.get("RECALL_HTTP_TOKEN"))
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE,
                         help="Fallback .env to read RECALL_HTTP_TOKEN from")
    args = parser.parse_args()

    token = args.token or read_env_file(args.env_file).get("RECALL_HTTP_TOKEN")
    if not token:
        print(f"error: no token found (checked --token, $RECALL_HTTP_TOKEN, {args.env_file})", file=sys.stderr)
        sys.exit(1)

    try:
        tool_args = json.loads(args.args_json)
    except json.JSONDecodeError as e:
        print(f"error: args_json is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    init_result, session_id = post(args.url, token, None, {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "recall-cli", "version": "0.1"}},
    })
    if "error" in init_result:
        print(f"initialize failed: {init_result['error']}", file=sys.stderr)
        sys.exit(1)

    result, _ = post(args.url, token, session_id, {
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": args.tool, "arguments": tool_args},
    })
    if "error" in result:
        print(f"tool call failed: {result['error']}", file=sys.stderr)
        sys.exit(1)

    r = result.get("result", {})
    if r.get("isError"):
        text = r.get("content", [{}])[0].get("text", r)
        print(f"tool returned an error: {text}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(r.get("structuredContent", r), indent=2))


if __name__ == "__main__":
    main()
