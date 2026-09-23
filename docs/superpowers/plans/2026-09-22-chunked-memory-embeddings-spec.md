# Chunked Memory Embeddings — Spec

## Problem

EmbeddingGemma's context window is 2048 tokens. `save_memory`/`update_memory`
embed `description + "\n\n" + body` as one vector; a memory whose combined
text exceeds that fails outright (as of the preflight-check fix already
shipped, a fast, clear rejection instead of the old opaque retried 500). The
caller then has to manually shorten or split the memory and retry — no data
is lost, but nothing is automatic, and a still-valid memory that's simply
long has no way to be fully searchable.

## Goal

A memory body of any length saves successfully, with the full text always
stored verbatim, and stays searchable end to end — without the caller ever
needing to manually split or shorten it.

## Design

Informed by how MemPalace and Hindsight (researched this session) handle
the same problem — see "Prior art" below.

1. **Split what gets embedded.** `description` (already documented as a
   one-line summary, so always well under the token budget) is embedded
   alone and becomes a memory's **primary** search vector. `body` is
   embedded as one or more **chunks**, each within the token budget, and
   becomes a **secondary** search signal.
2. **Chunk only when needed.** A body that already fits produces exactly one
   chunk — no behavior change for the overwhelming majority of memories.
   Splitting prefers paragraph boundaries, falling back to sentences, then
   words, so a chunk reads as a coherent unit rather than an arbitrary token
   window.
3. **Two-phase, deduplicated search.** A memory's final score is the better
   of its description-embedding score and its best single chunk's score.
   This is the piece that avoids the failure mode MemPalace's flat
   chunk-only search would have: without deduplication, a handful of very
   long memories could occupy several of the top-K slots with their own
   chunks and crowd out other, unrelated memories.
4. **Full-text storage is untouched by any of this.** `memories.body` is
   never truncated or rewritten by chunking — it already lives in a
   separate column from the embedding, so "no data loss" is structural, not
   something the chunking logic has to get right.
5. **Updates replace the whole chunk set, never patch it.** Reusing old
   chunk rows by index when a new body produces fewer chunks would leave a
   stale chunk from the previous version still matching in search — the
   same class of bug the existing "never leave a stale vector" merged
   re-embed already guards against for the single-vector case.
6. **Description and body now embed independently.** Today, changing either
   field re-embeds both, merged. Splitting them apart makes `update_memory`
   only recompute what actually changed — a smaller behavior footprint, not
   a bigger one.

## Prior art (this session's research)

- **MemPalace** chunks only when a fixed `chunk_size` is exceeded, and
  searches a compact per-topic "closet" layer first, resolving to specific
  chunks ("drawers") only on a hit — chunks are a drill-down layer, not the
  primary ranking surface. recall's description embedding plays the same
  role as a closet, using metadata (`description`) the tool already
  requires rather than a separately mined summary.
- **Hindsight** keeps original document text stored completely separately
  from whatever gets chunked/embedded from it, and does LLM-based
  consolidation so overlapping facts merge into one observation instead of
  accumulating — a stronger, root-cause answer to memories growing via
  repeated appends. Consolidation is out of scope for this plan (see
  below); recall already gets Hindsight's "verbatim storage is structural"
  property for free from its existing schema (`body` is a plain column
  untouched by `embedding`).

## Out of scope

- **Automatic summarization/consolidation** of appended updates
  (Hindsight-style). Recall's own memory-writing convention already asks
  the calling agent to condense growing memories by hand; automating that
  would need an LLM call inside `update_memory` to judge merge-vs-append,
  which is a separate, larger piece of work.
- **Staleness/relevance decay** (recency-weighted ranking, supersession
  links, write-time duplicate detection). A distinct problem — chunking is
  about *fitting* content into a vector, staleness is about *trusting* a
  vector once retrieved — covered by a separate plan.
- **Cross-memory digests/rollups** (clustering many small memories on one
  topic into a shared summary). MemPalace-adjacent idea, bigger scope than
  a single memory's own chunking.

## Compatibility

- No MCP tool schema changes — `save_memory`/`update_memory`/`search_memory`
  argument and result shapes are unchanged; this is an internal
  storage/ranking change.
- No new environment variables — reuses `RECALL_EMBEDDING_MAX_TOKENS`
  (already shipped) as the per-chunk token budget.
- Existing memories (pre-migration) keep working: `SearchMemories` degrades
  gracefully to description-only scoring for any memory with zero chunk
  rows. A `backfill-chunks` command re-embeds every existing memory under
  the new scheme; running it isn't required for correctness, only for full
  chunk-level searchability of pre-existing long memories.
