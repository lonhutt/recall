---
name: recall
description: Operate and troubleshoot Recall, Lon's self-hosted MCP memory service (Postgres + pgvector + a local llama.cpp embedding model, all containerized in this repo). Use this whenever recall, the memory stack, or its MCP tools (save_memory, search_memory, list_memories, update_memory, delete_memory, log_event, list_events) come up in the context of this project — checking whether it's healthy, bringing it up or down, viewing logs, rotating the HTTP bearer token, running schema migrations, or re-embedding after switching the embedding model. Also use it to call recall's tools directly over HTTP via the bundled script whenever the recall MCP server isn't connected in the current Claude Code session. Trigger even on indirect phrasing like "is recall running", "recall seems broken", "check my memories for X" while working in this repo, or "I changed the embedding model, does existing data need to migrate" — not just literal mentions of "recall" or "MCP". Also covers installing Recall from scratch on a machine that has never run it (asking for Docker group access, generating the bearer token, bringing the three containers up, registering the server in ~/.claude.json), so trigger on "set up recall", "install recall", "get recall running on this machine", or "recall isn't connected, can you wire it up".
---

# Operating Recall

Recall is a Go MCP server (`internal/`, `cmd/recall/`) backed by three
containers defined in `docker-compose.yml`: `postgres` (pgvector), `embeddings`
(a llama.cpp server running Google's EmbeddingGemma locally — no memory
content ever leaves the machine), and `recall` itself, which is the only one
of the three reachable from the host (`127.0.0.1:8092`, HTTP, bearer-token
authenticated). `postgres` and `embeddings` are compose-internal only —
that's intentional, not a bug, since `recall` is their sole consumer.

## Installing on a machine that has never run it

Only for a first-time setup; if `docker compose ps` already lists the three
services, skip ahead to "Operating the stack".

### 1. Check Docker access before anything else

```sh
docker info >/dev/null 2>&1 && echo ok
```

If that fails on permissions, the account isn't in the `docker` group in this
session. Adding it needs sudo, so ask the user to run it rather than trying it
yourself (they can prefix the command with `!` in the prompt to run it inline):

```sh
groups                              # is "docker" listed?
sudo usermod -aG docker "$USER"     # only if it isn't
```

The non-obvious part: `/etc/group` can already show the membership while the
running session still has stale groups; `groups` is what tells the two apart.
A reboot is what actually took effect here; `newgrp docker` only fixes the one
shell it runs in, which does the Claude Code session no good. So if `groups`
looks stale, ask them to reboot and pick this back up afterward.

### 2. Create .env

```sh
cp .env.example .env
openssl rand -hex 32   # paste it in as RECALL_HTTP_TOKEN=
```

`.env` is gitignored. Along with `~/.claude.json` it's one of only two places
the token lives, and the two have to match exactly or every call comes back 401.

### 3. Bring the stack up

```sh
docker compose up -d --build
docker compose ps
```

`embeddings` downloads EmbeddingGemma from Hugging Face on its first start, so
it can sit in "starting" for a minute or more; that's the model pull, not a
hang. `recall` waits on it and on `postgres` reporting healthy before it boots.
Migrations apply on start (`RECALL_MIGRATE_ON_START` defaults to true), so
there's no separate migrate step to run; `make migrate-up` couldn't reach the
compose Postgres anyway, since that service publishes no host port.

Once all three are healthy, check recall is really serving:

```sh
curl -sf http://127.0.0.1:8092/healthz && echo
python3 .claude/skills/recall/scripts/recall_cli.py list_memories '{"limit": 1}'
```

Do both. `/healthz` is unauthenticated, so it only proves the process is up;
the CLI call is what proves the token works.

### 4. Register it with Claude Code

This edits `~/.claude.json`, which is the user's global config and is about to
hold a bearer token, so ask first and back it up:

```sh
cp ~/.claude.json ~/.claude.json.bak
```

Then add this to the top-level `mcpServers` object, alongside whatever entries
are already there:

```json
"recall": {
  "type": "http",
  "url": "http://127.0.0.1:8092/mcp",
  "headers": {
    "Authorization": "Bearer <the RECALL_HTTP_TOKEN value from .env>"
  }
}
```

Nothing else in the file changes. `RECALL_TRANSPORT` stays `http` in
`docker-compose.yml`; the stdio transport still exists in the binary but
doesn't apply here, since a container isn't something Claude Code spawns.

### 5. Restart Claude Code

MCP servers connect at session start, so the seven tools won't show up until a
new session; nothing is broken in the meantime. Until the restart,
`recall_cli.py` is the way in (see "When they're not connected").

## When MCP tools are already connected

If `save_memory`, `search_memory`, `list_memories`, `update_memory`,
`delete_memory`, `log_event`, and `list_events` show up as available tools,
just use them directly — that's the normal path, and nothing in this skill
needs to intervene. A few conventions worth following when calling them:

- `project`: the absolute path of the project the memory relates to (e.g.
  `/home/lhutt/Projects/dcx`), not a short name — this is what `list_memories`
  and `search_memory`'s `project` filter match against.
- `agent`: `"claude-code"`.
- `type`: one of `user` / `feedback` / `project` / `reference` — same
  taxonomy the file-based memory system used, so the same judgment about
  what's worth saving applies (surprising, non-obvious, not derivable from
  code or git history — not routine facts).
- Before saving something that might already be known, `search_memory` or
  `list_memories` first — cheaper than a duplicate, and `update_memory` is
  the right tool if it turns out to already exist (patches only given
  fields, re-embeds only if description/body changed).

## When they're not connected

MCP servers connect at session start, so a config change to `~/.claude.json`
(or the stack just being down) means the tools won't show up until the next
session — or ever, if you're scripting outside Claude Code entirely. For
those cases, `scripts/recall_cli.py` does the same HTTP handshake
(`initialize` → capture `Mcp-Session-Id` → `tools/call`) that the MCP client
would, so you can call any tool from a shell:

```sh
python3 .claude/skills/recall/scripts/recall_cli.py list_memories '{"limit": 5}'
python3 .claude/skills/recall/scripts/recall_cli.py search_memory '{"query": "how does Lon like commits structured"}'
python3 .claude/skills/recall/scripts/recall_cli.py save_memory '{"type": "project", "slug": "x", "description": "d", "body": "b"}'
```

It reads `RECALL_HTTP_TOKEN` from `.env` by default (or `--token`/`$RECALL_HTTP_TOKEN`
to override), and gives a specific error for the two failure modes that
actually happen — 401 (token mismatch, likely rotated — see below) and
connection-refused (stack isn't up — see below).

## Operating the stack

```sh
docker compose up -d --build   # bring everything up (builds recall's image if code changed)
docker compose ps              # health of all three — "embeddings" can take ~30-60s on
                                # first start while it downloads EmbeddingGemma from HF
docker compose logs recall -f  # or postgres / embeddings
docker compose down            # stop everything (volumes persist)
docker compose exec postgres psql -U recall -d recall   # manual DB access; no port is
                                                          # published, this is the way in
```

Common failure signatures:
- **`recall_cli.py` says "could not reach ... is the stack up?"** — `docker compose up -d`.
- **401 from the CLI or from Claude Code's MCP connection** — `RECALL_HTTP_TOKEN` in `.env`
  doesn't match the `Authorization` header in `~/.claude.json`'s `recall` entry. This
  happens if one was updated without the other (see rotation below).
- **`embeddings` stuck "starting" for a long time** — check `docker compose logs embeddings`;
  it's downloading the model on first run, not actually stuck, unless the log stops moving.

## Rotating the HTTP token

```sh
openssl rand -hex 32              # generate a new one
# put it in .env as RECALL_HTTP_TOKEN=<new value>
docker compose up -d              # recreates the recall container with the new env
```

Then update the `Authorization: Bearer <token>` header in `~/.claude.json`'s
`recall` MCP entry to match, and restart Claude Code — the connection won't
pick up the change mid-session.

## Schema migrations

Ordinary migrations (`internal/store/postgres/migrations/*.sql`,
golang-migrate) apply automatically on container start via
`RECALL_MIGRATE_ON_START` (default true) — nothing extra to do.

A migration that **changes an existing column's shape** (like the embedding
dimension change from Voyage's 1024 to EmbeddingGemma's 768) is different:
pgvector can't cast a vector to a different dimension, so you can't just
`ALTER COLUMN ... TYPE vector(N)` against live data. The pattern used last
time (search Recall itself for `recall-local-embeddings-migration` — it's
saved as a memory, self-documenting) was two migrations with an
application-level step in between:

1. Migration N: drop the column, re-add it nullable at the new
   width/type, dropping any index on it.
2. Backfill every row at the application level — the trick is calling
   `update_memory` with each row's *own existing* `description`, which
   forces a re-embed without changing any content.
3. Migration N+1: `SET NOT NULL`, rebuild the index.

Do it in that order specifically — never write both migrations up front and
run them back-to-back, or you skip the backfill window and migration N+1's
`SET NOT NULL` will simply fail against the still-NULL rows (safe, but
means redoing the sequence correctly anyway).

## Changing the embedding model

If `RECALL_EMBEDDING_MODEL`/the `embeddings` container's model ever changes
again, existing memories' vectors are in the *old* model's embedding space —
cosine similarity between an old vector and a new query vector is
meaningless, even if the dimension happens to match. Existing data needs the
same backfill step from the migrations section above (re-embed every row via
`update_memory`), whether or not the dimension itself changed.

Two other things change with the model, neither of which is a data problem:

- **The query/document prompt prefixes are configurable, not hardcoded.**
  `RECALL_EMBEDDING_QUERY_PREFIX`/`RECALL_EMBEDDING_DOCUMENT_PREFIX`
  (`internal/config/config.go`, consumed by `internal/embeddings/llamacpp`)
  default to EmbeddingGemma's asymmetric-retrieval convention. A different
  model with its own convention (or none) just needs these env vars changed
  in `docker-compose.yml` — no Go code change, unlike before this was
  externalized.
- **A genuinely different provider still needs a new `Embedder`.** Prefixes
  aside, `internal/embeddings/llamacpp` is an HTTP client for one specific
  server shape (llama.cpp's OpenAI-compatible endpoint, no auth). A model
  served a completely different way (different API, needs an API key, etc.)
  needs a new package implementing the `Embedder` interface
  (`internal/tools/deps.go`) and rewiring in `main.go` — the same seam the
  Voyage→llama.cpp swap used.
- **`embedding_model` is stored per row but not enforced anywhere.**
  `search_memory` ranks every row by cosine distance with no check on which
  model produced which row's vector — a partial backfill doesn't error, it
  silently degrades ranking quality by comparing incompatible vector spaces.
  This is why the backfill in the migrations section has to run to
  completion before search results are trustworthy again, not just started.
