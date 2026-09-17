# Recall

An MCP server for persisted memory, backed by Postgres + pgvector. Any
MCP-capable agent can save, search, and list structured memory notes, and
log/list timestamped episodic events. Embeddings are generated locally via a
llama.cpp server running Google's EmbeddingGemma; no memory content or
search query ever leaves the machine.

## Requirements

- Docker (runs Postgres, the local embedding server, and Recall itself as
  three services in `docker-compose.yml`)
- Go 1.27+ (only needed for native development/testing; the primary way to
  run Recall is fully containerized)

## Running

```sh
cp .env.example .env
# generate a bearer token for the HTTP endpoint and put it in .env:
openssl rand -hex 32   # -> RECALL_HTTP_TOKEN=<paste here>

docker compose up -d --build
```

This builds and starts all three services: `postgres`, `embeddings`, and
`recall`. `recall` waits for the other two to report healthy, then runs
migrations automatically on startup (unless `RECALL_MIGRATE_ON_START=false`)
and serves MCP over HTTP on `127.0.0.1:8092` (`/mcp`, bearer-token
authenticated; `/healthz` is unauthenticated liveness, which is what the
container's healthcheck hits). `postgres` and `embeddings`
publish no host ports at all; `recall` is their only consumer, and reaches
them over the internal compose network by service name. The embedding
server downloads EmbeddingGemma from Hugging Face on its first start
(cached in the `llama-embed-cache` volume after that, so it's fully offline
from then on).

Native execution (`make build && make run`, stdio transport, the default
when `RECALL_TRANSPORT` is unset) still works for quick local testing, but
it needs its own reachable Postgres/embeddings; it can't reach the
compose-managed ones once their ports aren't published.

## Testing

```sh
make test              # fast suite: models, embedding client, tool handlers, MCP wiring, no Docker needed
make test-integration   # + store and Postgres-backed tests, requires Docker (spins up its own ephemeral Postgres via testcontainers-go, not the compose one)
```

## Wiring into an MCP client (e.g. Claude Code)

```json
{
  "mcpServers": {
    "recall": {
      "type": "http",
      "url": "http://127.0.0.1:8092/mcp",
      "headers": {
        "Authorization": "Bearer <RECALL_HTTP_TOKEN from your .env>"
      }
    }
  }
}
```

## Tools

The following are exposed as MCP tools:

| Tool | Purpose |
|---|---|
| `save_memory` | Create a structured note (type, slug, description, body, tags) |
| `update_memory` | Patch an existing note by slug (only given fields change; re-embeds if description/body changed) |
| `search_memory` | Semantic search over notes, with optional type/project/agent/tag filters |
| `list_memories` | Browse notes by metadata filters, no embedding call |
| `delete_memory` | Delete a note by slug |
| `log_event` | Record a timestamped episodic event |
| `list_events` | List episodic events by metadata/date-range filters |

## Operational notes

- Embedding dimension is fixed at `models.EmbeddingDimension` (768, matching
  EmbeddingGemma's native output). Changing the embedding model or
  dimension requires a new migration (`ALTER TABLE ... ALTER COLUMN
  embedding TYPE vector(N)`) and re-embedding every existing row; there's
  no tooling for that yet.
- `slug` is globally unique across all memories (one flat namespace, not
  scoped per project).
- The embedding server is a separate `llama-server` instance from any
  other llama.cpp server you might already be running natively (for chat,
  say); it runs in `--embedding`-only mode and can't serve both use cases
  at once.
- The query/document asymmetric-retrieval prompt prefixes are configurable
  (`RECALL_EMBEDDING_QUERY_PREFIX`/`RECALL_EMBEDDING_DOCUMENT_PREFIX`,
  defaulting to EmbeddingGemma's convention), so a different model with a
  different convention doesn't need a code change. A genuinely different
  provider (different request/response shape, auth, etc.) still needs a
  new `Embedder` implementation; same as the Voyage to llama.cpp swap.
