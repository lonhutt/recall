# Chunked Memory Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Memories of any body length save successfully, stay fully searchable, and never lose data — by embedding `description` alone as the primary search vector and splitting `body` into token-budget-sized chunks as a secondary, deduplicated search layer.

**Architecture:** A new `memory_chunks` table (one row per chunk, FK to `memories`, its own HNSW index) sits alongside the existing `memories` table, whose `embedding` column now holds the description embedding only. `save_memory`/`update_memory` compute both independently via a shared helper; writes are transactional so a memory and its chunk set are never inconsistent. `SearchMemories` becomes two queries merged in Go: best description match, best chunk match per memory, final score is the max of the two, deduplicated to one row per memory.

**Tech Stack:** Go 1.27.1, Postgres 17 + pgvector (`pgx/v5`), llama.cpp embeddings server (EmbeddingGemma-300M), `golang-migrate`, MCP Go SDK. No new dependencies.

**Spec:** `docs/superpowers/plans/2026-09-22-chunked-memory-embeddings-spec.md`

## Global Constraints

- No new dependencies — everything below uses `pgx/v5`, `pgvector-go`, and the standard library.
- `memories.body` is never truncated, rewritten, or otherwise touched by chunking logic — full verbatim storage is structural, not best-effort.
- Any write that touches `memories` and `memory_chunks` together (create, or update that changes `Body`) is one Postgres transaction — never a memory row with a partial or stale chunk set.
- No MCP tool schema changes: `SaveMemoryArgs`, `UpdateMemoryArgs`, `SearchMemoryArgs`, and all `*Result` types are unchanged.
- No new environment variables — the per-chunk token budget reuses `RECALL_EMBEDDING_MAX_TOKENS` (`cfg.EmbeddingMaxTokens`, already shipped).
- Follow existing test conventions exactly: `internal/tools` uses `fakeStore`/`fakeEmbedder` unit tests; `internal/store/postgres` uses `//go:build integration` + testcontainers, skipped when Docker isn't reachable.
- All new code must pass a lazy-senior-dev review pass (YAGNI, reuse before writing, no speculative abstractions, shortest correct diff) — reviewers apply this in addition to spec compliance.

## Review Focus

- A body that already fits in one chunk must produce byte-identical single-chunk behavior — no regression for the common case (Task 1, Task 5).
- A memory with zero chunk rows (pre-backfill data, or the window before a migration runs) must not error or panic `SearchMemories` — it should just rank on its description score alone (Task 4).
- An `update_memory` call that only changes unrelated fields (e.g. `tags`) must not re-embed anything or touch the chunk table (Task 3, existing behavior preserved).
- A body update must fully replace the old chunk set — a shorter new body must never leave a stale chunk from the previous, longer version still matching in search (Task 3).
- A single indivisible oversized token run (one giant word/blob with no whitespace or punctuation to split on) must not hang the chunker in infinite recursion (Task 1).

---

## Task 1: `internal/chunking` package

**Files:**
- Create: `internal/chunking/chunk.go`
- Test: `internal/chunking/chunk_test.go`

**Interfaces:**
- Consumes: nothing (pure package, no dependency on this repo's other packages)
- Produces: `chunking.TokenCounter` (`func(ctx context.Context, text string) (int, error)`), `chunking.Chunk(ctx context.Context, text string, maxTokens int, counter TokenCounter) ([]string, error)` — used by Task 5's `internal/tools/embed.go`

- [ ] **Step 1: Write the failing tests**

```go
// internal/chunking/chunk_test.go
package chunking

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// wordCounter is a fake TokenCounter for tests: one token per word, so
// expected outputs are easy to reason about without a real tokenizer.
func wordCounter(_ context.Context, text string) (int, error) {
	return len(strings.Fields(text)), nil
}

func TestChunkReturnsTextUnchangedWhenWithinBudget(t *testing.T) {
	got, err := Chunk(context.Background(), "one two three", 10, wordCounter)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(got) != 1 || got[0] != "one two three" {
		t.Errorf("Chunk() = %v, want a single unchanged chunk", got)
	}
}

func TestChunkSplitsOnParagraphBoundariesWhenOverBudget(t *testing.T) {
	text := "one two three\n\nfour five six\n\nseven eight nine"
	got, err := Chunk(context.Background(), text, 6, wordCounter)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("Chunk() returned %d chunks, want 2 (paragraphs packed two-then-one): %v", len(got), got)
	}
	if got[0] != "one two three\n\nfour five six" {
		t.Errorf("chunk[0] = %q, want the first two paragraphs packed together", got[0])
	}
	if got[1] != "seven eight nine" {
		t.Errorf("chunk[1] = %q, want the third paragraph alone", got[1])
	}
}

func TestChunkEveryPieceFitsWithinMaxTokens(t *testing.T) {
	text := strings.Repeat("word ", 500)
	const maxTokens = 20
	got, err := Chunk(context.Background(), text, maxTokens, wordCounter)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	for i, c := range got {
		n, _ := wordCounter(context.Background(), c)
		if n > maxTokens {
			t.Errorf("chunk[%d] has %d tokens, want <= %d", i, n, maxTokens)
		}
	}
}

func TestChunkFallsBackToSentencesForAnOversizedParagraph(t *testing.T) {
	text := "one two. three four. five six. seven eight."
	got, err := Chunk(context.Background(), text, 4, wordCounter)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	for i, c := range got {
		if n, _ := wordCounter(context.Background(), c); n > 4 {
			t.Errorf("chunk[%d] = %q has %d tokens, want <= 4", i, c, n)
		}
	}
	rejoined := strings.Join(got, " ")
	for _, want := range []string{"one two.", "three four.", "five six.", "seven eight."} {
		if !strings.Contains(rejoined, want) {
			t.Errorf("rejoined chunks %q missing sentence %q", rejoined, want)
		}
	}
}

func TestChunkFallsBackToWordsForAnOversizedSentence(t *testing.T) {
	longRun := strings.TrimSpace(strings.Repeat("word ", 20))
	text := "intro. " + longRun
	const maxTokens = 5
	got, err := Chunk(context.Background(), text, maxTokens, wordCounter)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(got) < 2 {
		t.Fatalf("Chunk() = %v, want the long word run split into multiple chunks", got)
	}
	for i, c := range got {
		if n, _ := wordCounter(context.Background(), c); n > maxTokens {
			t.Errorf("chunk[%d] = %q has more than %d tokens", i, c, maxTokens)
		}
	}
}

func TestChunkTerminatesOnAnIndivisibleOversizedWord(t *testing.T) {
	text := strings.Repeat("x", 50)
	got, err := Chunk(context.Background(), text, 1, wordCounter)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(got) != 1 || got[0] != text {
		t.Errorf("Chunk() = %v, want the single indivisible run accepted as one oversized chunk", got)
	}
}

// A paragraph or sentence with no '.'/'!'/'?' — a bullet-style line, a
// run-on clause, a URL-heavy line — must still fall through to word-level
// splitting rather than being treated as indivisible just because the
// sentence-fallback tier found no punctuation to split on.
func TestChunkFallsBackToWordsWhenParagraphHasNoSentencePunctuation(t *testing.T) {
	text := strings.TrimSpace(strings.Repeat("word ", 20))
	const maxTokens = 5
	got, err := Chunk(context.Background(), text, maxTokens, wordCounter)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(got) < 2 {
		t.Fatalf("Chunk() = %v, want the punctuation-free run split by word", got)
	}
	for i, c := range got {
		if n, _ := wordCounter(context.Background(), c); n > maxTokens {
			t.Errorf("chunk[%d] = %q has %d tokens, want <= %d", i, c, n, maxTokens)
		}
	}
}

func TestChunkPropagatesCounterError(t *testing.T) {
	boom := errors.New("boom")
	failing := func(context.Context, string) (int, error) { return 0, boom }
	if _, err := Chunk(context.Background(), "hello", 10, failing); !errors.Is(err, boom) {
		t.Errorf("Chunk() error = %v, want it to wrap %v", err, boom)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./internal/chunking/... -v`
Expected: FAIL — build fails, `Chunk` and `TokenCounter` undefined (package doesn't exist yet).

- [ ] **Step 3: Write the implementation**

```go
// internal/chunking/chunk.go

// Package chunking splits text that's too long for an embedding model's
// context window into pieces that each fit, preferring paragraph
// boundaries, then sentences, then words, so a piece reads as a coherent
// unit rather than an arbitrary token window.
package chunking

import (
	"context"
	"fmt"
	"strings"
)

// TokenCounter counts how many tokens text will produce once embedded,
// e.g. (*llamacpp.Client).CountTokens.
type TokenCounter func(ctx context.Context, text string) (int, error)

// Chunk splits text into pieces that each fit within maxTokens according to
// counter, so every piece can be embedded on its own. Returns a
// single-element slice containing text unchanged when it already fits.
func Chunk(ctx context.Context, text string, maxTokens int, counter TokenCounter) ([]string, error) {
	if maxTokens <= 0 {
		return nil, fmt.Errorf("maxTokens must be positive, got %d", maxTokens)
	}
	n, err := counter(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("counting tokens: %w", err)
	}
	if n <= maxTokens {
		return []string{text}, nil
	}

	paragraphs := splitNonEmpty(text, "\n\n")
	return pack(ctx, paragraphs, "\n\n", maxTokens, counter, splitSentences)
}

// pack greedily groups pieces into chunks joined by sep, each within
// maxTokens. A piece that alone exceeds maxTokens is split further (via
// fallback, or word-splitting if fallback is nil) before being packed.
func pack(ctx context.Context, pieces []string, sep string, maxTokens int, counter TokenCounter, fallback func(string) []string) ([]string, error) {
	var chunks []string
	var current []string
	currentTokens := 0

	flush := func() {
		if len(current) > 0 {
			chunks = append(chunks, strings.Join(current, sep))
			current = nil
			currentTokens = 0
		}
	}

	for _, piece := range pieces {
		pieceTokens, err := counter(ctx, piece)
		if err != nil {
			return nil, fmt.Errorf("counting tokens: %w", err)
		}

		if pieceTokens > maxTokens {
			flush()
			// Try this tier's fallback first; if it finds no delimiter
			// (returns the piece unchanged — e.g. no sentence-ending
			// punctuation), fall through to word-splitting before
			// declaring the piece indivisible. Conflating "this fallback
			// found nothing" with "truly indivisible" would skip
			// word-splitting even when the piece has plenty of spaces to
			// split on.
			subPieces := fallbackOrUnchanged(piece, fallback)
			if len(subPieces) == 1 && subPieces[0] == piece {
				subPieces = splitNonEmpty(piece, " ")
			}
			// A single indivisible run (e.g. one giant word with no
			// spaces) can't be split further even by word; accept it as
			// one oversized chunk rather than recursing forever. Embed
			// will still reject it if it's truly too big to embed — an
			// honest failure, not silent data loss, since nothing here
			// drops text.
			if len(subPieces) == 1 && subPieces[0] == piece {
				chunks = append(chunks, piece)
				continue
			}
			sub, err := pack(ctx, subPieces, " ", maxTokens, counter, nil)
			if err != nil {
				return nil, err
			}
			chunks = append(chunks, sub...)
			continue
		}

		sepTokens := 0
		if len(current) > 0 {
			sepTokens, err = counter(ctx, sep)
			if err != nil {
				return nil, fmt.Errorf("counting tokens: %w", err)
			}
		}
		if len(current) > 0 && currentTokens+sepTokens+pieceTokens > maxTokens {
			flush()
		}
		current = append(current, piece)
		currentTokens += sepTokens + pieceTokens
	}
	flush()
	return chunks, nil
}

// fallbackOrUnchanged applies fallback to piece if given, returning
// []string{piece} unchanged when fallback is nil or found no delimiter to
// split on (so the caller can detect "no progress" uniformly and decide
// whether to try the next fallback tier).
func fallbackOrUnchanged(piece string, fallback func(string) []string) []string {
	if fallback == nil {
		return []string{piece}
	}
	return fallback(piece)
}

func splitNonEmpty(text, sep string) []string {
	var out []string
	for _, p := range strings.Split(text, sep) {
		if strings.TrimSpace(p) != "" {
			out = append(out, p)
		}
	}
	return out
}

// splitSentences breaks text after '.', '!', or '?' followed by whitespace.
// It's a heuristic, not a full sentence tokenizer: worst case it over- or
// under-splits by a few words, and pack() still enforces the token budget
// regardless of where the boundaries land.
func splitSentences(text string) []string {
	var out []string
	start := 0
	for i := 0; i < len(text); i++ {
		c := text[i]
		if (c == '.' || c == '!' || c == '?') && i+1 < len(text) && (text[i+1] == ' ' || text[i+1] == '\n') {
			if piece := strings.TrimSpace(text[start : i+1]); piece != "" {
				out = append(out, piece)
			}
			start = i + 1
		}
	}
	if rest := strings.TrimSpace(text[start:]); rest != "" {
		out = append(out, rest)
	}
	return out
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./internal/chunking/... -v`
Expected: PASS, all 7 tests.

- [ ] **Step 5: Commit**

```bash
git add internal/chunking
git commit -m "add token-aware text chunking package"
```

---

## Task 2: Expose `CountTokens` on the `Embedder` interface

**Files:**
- Modify: `internal/tools/deps.go`
- Modify: `internal/tools/fakes_test.go`

**Interfaces:**
- Consumes: `(*llamacpp.Client).CountTokens` already exists (shipped in the preflight-check fix) and satisfies the new interface method with no changes to `internal/embeddings/llamacpp`.
- Produces: `Embedder` interface gains `CountTokens(ctx context.Context, text string) (int, error)`; `fakeEmbedder` gains a matching method plus per-call history (`allCalls`) for tests that need to distinguish multiple `Embed` calls in one tool invocation; `fakeStore` gains `lastChunks []postgres.MemoryChunk` for tests that need to inspect what chunks a save/update produced.

- [ ] **Step 1: Write the failing test**

```go
// append to internal/tools/fakes_test.go — this alone won't compile until
// Step 3, which is the point: it proves the interface widening is real.
func TestFakeEmbedderSatisfiesWidenedEmbedderInterface(t *testing.T) {
	var _ Embedder = (*fakeEmbedder)(nil)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go vet ./internal/tools/...`
Expected: FAIL with a "missing method CountTokens" style error once `deps.go`'s interface is widened but the fake isn't updated yet. Do the interface change and the fake update together in Step 3, then use this vet run purely to confirm Step 1's test would have caught the gap: temporarily widen only `deps.go` first, run `go vet`, observe the failure, then proceed to update the fake.

- [ ] **Step 3: Widen the interface and update the fakes**

```go
// internal/tools/deps.go — change the Embedder interface
// Embedder generates text embeddings. Satisfied by *llamacpp.Client.
type Embedder interface {
	Embed(ctx context.Context, texts []string, kind embeddings.InputType) ([][]float32, error)
	CountTokens(ctx context.Context, text string) (int, error)
}
```

```go
// internal/tools/fakes_test.go — add "strings" to imports,
// add lastChunks to fakeStore, add CountTokens + allCalls to fakeEmbedder

// add to the fakeStore struct's field list:
	lastChunks []postgres.MemoryChunk

// in fakeStore.CreateMemory, right after the createErr check:
	f.lastChunks = m.Chunks

// in fakeStore.UpdateMemory, right after the updateErr check:
	if p.Chunks != nil {
		f.lastChunks = *p.Chunks
	}

// embedCall records one Embed invocation, for tests that need to assert on
// more than just the most recent call (e.g. a tool call that embeds a
// description and then, separately, one or more body chunks).
type embedCall struct {
	texts []string
	kind  embeddings.InputType
}

// fakeEmbedder is an Embedder double returning a fixed vector (or error) per call.
type fakeEmbedder struct {
	vector        []float32
	err           error
	lastKind      embeddings.InputType
	lastTexts     []string
	calls         int
	allCalls      []embedCall
	countTokensFn func(text string) (int, error)
}

func (f *fakeEmbedder) Embed(_ context.Context, texts []string, kind embeddings.InputType) ([][]float32, error) {
	f.calls++
	f.lastTexts = texts
	f.lastKind = kind
	f.allCalls = append(f.allCalls, embedCall{texts: texts, kind: kind})
	if f.err != nil {
		return nil, f.err
	}
	vectors := make([][]float32, len(texts))
	for i := range texts {
		vectors[i] = f.vector
	}
	return vectors, nil
}

// CountTokens is a fake tokenizer for tests: one token per word, so tests
// needing deterministic chunk boundaries can reason about expected output
// without a real tokenizer. countTokensFn overrides this when a test needs
// specific counts.
func (f *fakeEmbedder) CountTokens(_ context.Context, text string) (int, error) {
	if f.countTokensFn != nil {
		return f.countTokensFn(text)
	}
	return len(strings.Fields(text)), nil
}
```

Note: `fakes_test.go` already imports `"github.com/lonhutt/recall/internal/store/postgres"`; only `"strings"` needs adding.

- [ ] **Step 4: Run tests to verify they pass**

Run: `go build ./internal/tools/... && go test ./internal/tools/... -run TestFakeEmbedderSatisfiesWidenedEmbedderInterface -v`
Expected: PASS. Then run the full package to confirm nothing else broke:

Run: `go test ./internal/tools/... -v`
Expected: PASS (no existing test references `CountTokens` or `allCalls` yet, so all prior tests are unaffected by this step alone).

- [ ] **Step 5: Commit**

```bash
git add internal/tools/deps.go internal/tools/fakes_test.go
git commit -m "expose CountTokens on the Embedder interface"
```

---

## Task 3: `memory_chunks` schema + transactional chunk writes

**Files:**
- Create: `internal/store/postgres/migrations/000006_create_memory_chunks.up.sql`
- Create: `internal/store/postgres/migrations/000006_create_memory_chunks.down.sql`
- Create: `internal/store/postgres/chunks.go`
- Modify: `internal/store/postgres/notes.go`
- Test: `internal/store/postgres/chunks_test.go`

**Interfaces:**
- Consumes: `newTestStore(t)` and `vector(...)` from `internal/store/postgres/notes_test.go` (existing test helpers, same package).
- Produces: `postgres.MemoryChunk{ChunkIndex int, Text string, Embedding []float32}`; `NewMemory.Chunks []MemoryChunk`; `MemoryPatch.Chunks *[]MemoryChunk`; `(*Store).CountMemoryChunks(ctx, slug) (int, error)` — used by Task 4's search tests and Task 6's backfill.

- [ ] **Step 1: Write the failing tests**

```go
// internal/store/postgres/chunks_test.go
//go:build integration

package postgres_test

import (
	"context"
	"testing"

	"github.com/lonhutt/recall/internal/store/postgres"
)

func TestCreateMemoryPersistsChunks(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	_, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "chunked", Description: "d", Body: "b",
		Embedding: vector(1, 0), EmbeddingModel: "embeddinggemma",
		Chunks: []postgres.MemoryChunk{
			{ChunkIndex: 0, Text: "first chunk", Embedding: vector(1, 0)},
			{ChunkIndex: 1, Text: "second chunk", Embedding: vector(0, 1)},
		},
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	count, err := store.CountMemoryChunks(ctx, "chunked")
	if err != nil {
		t.Fatalf("CountMemoryChunks: %v", err)
	}
	if count != 2 {
		t.Errorf("chunk count = %d, want 2", count)
	}
}

func TestDeleteMemoryCascadesToChunks(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	_, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "cascaded", Description: "d", Body: "b",
		Embedding: vector(1, 0), EmbeddingModel: "embeddinggemma",
		Chunks: []postgres.MemoryChunk{{ChunkIndex: 0, Text: "only chunk", Embedding: vector(1, 0)}},
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	if err := store.DeleteMemory(ctx, "cascaded"); err != nil {
		t.Fatalf("DeleteMemory: %v", err)
	}

	// CountMemoryChunks joins through memories, so once the memory row is
	// gone the join returns zero rows, not an error.
	count, err := store.CountMemoryChunks(ctx, "cascaded")
	if err != nil {
		t.Fatalf("CountMemoryChunks: %v", err)
	}
	if count != 0 {
		t.Errorf("chunk count after delete = %d, want 0", count)
	}
}

func TestUpdateMemoryReplacesChunksWholesale(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	_, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "replaced", Description: "d", Body: "b",
		Embedding: vector(1, 0), EmbeddingModel: "embeddinggemma",
		Chunks: []postgres.MemoryChunk{
			{ChunkIndex: 0, Text: "old chunk one", Embedding: vector(1, 0)},
			{ChunkIndex: 1, Text: "old chunk two", Embedding: vector(0, 1)},
			{ChunkIndex: 2, Text: "old chunk three", Embedding: vector(1, 1)},
		},
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	newBody := "shorter body"
	newChunks := []postgres.MemoryChunk{{ChunkIndex: 0, Text: newBody, Embedding: vector(0, 1)}}
	if _, err := store.UpdateMemory(ctx, "replaced", postgres.MemoryPatch{Body: &newBody, Chunks: &newChunks}); err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}

	count, err := store.CountMemoryChunks(ctx, "replaced")
	if err != nil {
		t.Fatalf("CountMemoryChunks: %v", err)
	}
	if count != 1 {
		t.Errorf("chunk count after replace = %d, want 1 (old chunks 1-2 must not survive)", count)
	}
}

func TestUpdateMemoryLeavesChunksUntouchedWhenPatchOmitsThem(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	_, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "untouched", Description: "d", Body: "b",
		Embedding: vector(1, 0), EmbeddingModel: "embeddinggemma",
		Chunks: []postgres.MemoryChunk{{ChunkIndex: 0, Text: "chunk", Embedding: vector(1, 0)}},
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	newProject := "recall"
	if _, err := store.UpdateMemory(ctx, "untouched", postgres.MemoryPatch{Project: &newProject}); err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}

	count, err := store.CountMemoryChunks(ctx, "untouched")
	if err != nil {
		t.Fatalf("CountMemoryChunks: %v", err)
	}
	if count != 1 {
		t.Errorf("chunk count = %d, want 1 (a patch that doesn't touch Chunks must leave them alone)", count)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test -tags integration ./internal/store/postgres/... -run 'TestCreateMemoryPersistsChunks|TestDeleteMemoryCascadesToChunks|TestUpdateMemoryReplacesChunksWholesale|TestUpdateMemoryLeavesChunksUntouchedWhenPatchOmitsThem' -v`
Expected: FAIL to build — `postgres.MemoryChunk` undefined, `NewMemory.Chunks` and `MemoryPatch.Chunks` fields don't exist, `CountMemoryChunks` undefined. (If Docker isn't reachable these tests will otherwise skip rather than fail — verify Docker is running first with `docker ps`.)

- [ ] **Step 3: Write the migration and store implementation**

```sql
-- internal/store/postgres/migrations/000006_create_memory_chunks.up.sql
CREATE TABLE memory_chunks (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id   uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    chunk_index int NOT NULL,
    chunk_text  text NOT NULL,
    embedding   vector(768) NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (memory_id, chunk_index)
);

CREATE INDEX memory_chunks_memory_id_idx ON memory_chunks (memory_id);
CREATE INDEX memory_chunks_embedding_idx ON memory_chunks USING hnsw (embedding vector_cosine_ops);
```

```sql
-- internal/store/postgres/migrations/000006_create_memory_chunks.down.sql
DROP TABLE memory_chunks;
```

```go
// internal/store/postgres/chunks.go
package postgres

import (
	"context"

	"github.com/jackc/pgx/v5"
	pgvector "github.com/pgvector/pgvector-go"
)

// MemoryChunk is one piece of a memory's body embedded on its own, for a
// body too long to fit the embedding model's context window in a single
// vector. ChunkIndex is 0-based and defines chunk order; a plain memory
// with a short body still gets exactly one chunk.
type MemoryChunk struct {
	ChunkIndex int
	Text       string
	Embedding  []float32
}

// replaceMemoryChunks deletes memoryID's existing chunks and inserts chunks
// in their place, as part of tx: a chunk set is always replaced wholesale,
// never patched, so an update can never leave a stale chunk from the
// previous body version sitting alongside the new ones.
func replaceMemoryChunks(ctx context.Context, tx pgx.Tx, memoryID string, chunks []MemoryChunk) error {
	if _, err := tx.Exec(ctx, `DELETE FROM memory_chunks WHERE memory_id = $1`, memoryID); err != nil {
		return err
	}
	for _, c := range chunks {
		if _, err := tx.Exec(ctx,
			`INSERT INTO memory_chunks (memory_id, chunk_index, chunk_text, embedding) VALUES ($1, $2, $3, $4)`,
			memoryID, c.ChunkIndex, c.Text, pgvector.NewVector(c.Embedding),
		); err != nil {
			return err
		}
	}
	return nil
}

// CountMemoryChunks returns how many chunk rows exist for slug's memory.
// Mainly for tests and diagnostics — production code only ever consumes
// chunks indirectly, through what SearchMemories returns.
func (s *Store) CountMemoryChunks(ctx context.Context, slug string) (int, error) {
	var count int
	err := s.pool.QueryRow(ctx,
		`SELECT count(*) FROM memory_chunks mc JOIN memories m ON m.id = mc.memory_id WHERE m.slug = $1`,
		slug,
	).Scan(&count)
	return count, err
}
```

```go
// internal/store/postgres/notes.go — add Chunks to both structs
type NewMemory struct {
	Type           string
	Slug           string
	Description    string
	Body           string
	Project        string
	Agent          string
	Tags           []string
	Embedding      []float32
	EmbeddingModel string
	Chunks         []MemoryChunk
}

type MemoryPatch struct {
	Type           *string
	Description    *string
	Body           *string
	Project        *string
	Agent          *string
	Tags           *[]string
	Embedding      []float32
	EmbeddingModel *string
	Chunks         *[]MemoryChunk
}
```

```go
// internal/store/postgres/notes.go — replace CreateMemory and UpdateMemory
func (s *Store) CreateMemory(ctx context.Context, m NewMemory) (*models.Memory, error) {
	tags := m.Tags
	if tags == nil {
		tags = []string{}
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx)

	row := tx.QueryRow(ctx, `
		INSERT INTO memories (type, slug, description, body, project, agent, tags, embedding, embedding_model)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		RETURNING `+memoryColumns,
		m.Type, m.Slug, m.Description, m.Body, nullableString(m.Project), nullableString(m.Agent),
		tags, pgvector.NewVector(m.Embedding), m.EmbeddingModel)

	mem, err := scanMemory(row)
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" {
			return nil, models.ErrDuplicateSlug
		}
		return nil, err
	}

	if err := replaceMemoryChunks(ctx, tx, mem.ID, m.Chunks); err != nil {
		return nil, err
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return mem, nil
}

func (s *Store) UpdateMemory(ctx context.Context, slug string, p MemoryPatch) (*models.Memory, error) {
	var sets []string
	var args []any
	set := func(col string, v any) {
		args = append(args, v)
		sets = append(sets, fmt.Sprintf("%s = $%d", col, len(args)))
	}

	if p.Type != nil {
		set("type", *p.Type)
	}
	if p.Description != nil {
		set("description", *p.Description)
	}
	if p.Body != nil {
		set("body", *p.Body)
	}
	if p.Project != nil {
		set("project", nullableString(*p.Project))
	}
	if p.Agent != nil {
		set("agent", nullableString(*p.Agent))
	}
	if p.Tags != nil {
		set("tags", *p.Tags)
	}
	if p.Embedding != nil {
		set("embedding", pgvector.NewVector(p.Embedding))
	}
	if p.EmbeddingModel != nil {
		set("embedding_model", *p.EmbeddingModel)
	}
	sets = append(sets, "updated_at = now()")

	args = append(args, slug)
	query := fmt.Sprintf(`UPDATE memories SET %s WHERE slug = $%d RETURNING %s`,
		strings.Join(sets, ", "), len(args), memoryColumns)

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx)

	mem, err := scanMemory(tx.QueryRow(ctx, query, args...))
	if err != nil {
		return nil, err
	}

	if p.Chunks != nil {
		if err := replaceMemoryChunks(ctx, tx, mem.ID, *p.Chunks); err != nil {
			return nil, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return mem, nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test -tags integration ./internal/store/postgres/... -v`
Expected: PASS for all postgres integration tests, including the 4 new ones and every pre-existing one (`TestCreateAndGetMemory`, `TestSearchMemoriesOrdersByCosineSimilarity`, etc. — none of them set `Chunks`, so they exercise the nil-slice/no-op path of `replaceMemoryChunks`).

Also run the non-integration suite to confirm nothing outside postgres broke:

Run: `go build ./... && go test ./...`
Expected: PASS (the tools package doesn't call `CreateMemory`/`UpdateMemory` directly yet in a way that would break from the new `Chunks` field, since Go struct literals not setting a new field just leave it at zero value).

- [ ] **Step 5: Commit**

```bash
git add internal/store/postgres/migrations/000006_create_memory_chunks.up.sql internal/store/postgres/migrations/000006_create_memory_chunks.down.sql internal/store/postgres/chunks.go internal/store/postgres/notes.go internal/store/postgres/chunks_test.go
git commit -m "add memory_chunks table and transactional chunk writes"
```

---

## Task 4: Two-phase `SearchMemories`

**Files:**
- Modify: `internal/store/postgres/search.go`
- Test: `internal/store/postgres/search_test.go`

**Interfaces:**
- Consumes: `buildFilter` (`internal/store/postgres/filter.go`, unchanged), `memoryColumns`/`scanMemory` (`notes.go`, unchanged).
- Produces: `SearchMemories`'s signature and `ScoredMemory` type are unchanged — this task only changes ranking behavior, not the API.

- [ ] **Step 1: Write the failing tests**

```go
// append to internal/store/postgres/search_test.go
func TestSearchMemoriesFindsMatchViaBodyChunkNotJustDescription(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	// "long-memory" only matches the query through its second chunk; its
	// description embedding points elsewhere entirely.
	_, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "long-memory", Description: "unrelated topic", Body: "irrelevant intro, then the real content",
		Embedding: vector(0, 1), EmbeddingModel: "embeddinggemma",
		Chunks: []postgres.MemoryChunk{
			{ChunkIndex: 0, Text: "irrelevant intro", Embedding: vector(0, 1)},
			{ChunkIndex: 1, Text: "the real content", Embedding: vector(1, 0)},
		},
	})
	if err != nil {
		t.Fatalf("CreateMemory(long-memory): %v", err)
	}
	_, err = store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "short-memory", Description: "also unrelated", Body: "b",
		Embedding: vector(0, 1), EmbeddingModel: "embeddinggemma",
		Chunks: []postgres.MemoryChunk{{ChunkIndex: 0, Text: "b", Embedding: vector(0, 1)}},
	})
	if err != nil {
		t.Fatalf("CreateMemory(short-memory): %v", err)
	}

	results, err := store.SearchMemories(ctx, vector(1, 0), postgres.MemoryFilter{}, 10)
	if err != nil {
		t.Fatalf("SearchMemories: %v", err)
	}
	if len(results) == 0 || results[0].Slug != "long-memory" {
		t.Fatalf("top result = %+v, want long-memory to rank first via its body chunk", results)
	}
}

func TestSearchMemoriesDeduplicatesMultipleChunkHitsToOneRowPerMemory(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	_, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "multi-chunk", Description: "d", Body: "b",
		Embedding: vector(0, 1), EmbeddingModel: "embeddinggemma",
		Chunks: []postgres.MemoryChunk{
			{ChunkIndex: 0, Text: "chunk one", Embedding: vector(1, 0)},
			{ChunkIndex: 1, Text: "chunk two", Embedding: vector(0.9, 0.1)},
			{ChunkIndex: 2, Text: "chunk three", Embedding: vector(0.8, 0.2)},
		},
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	results, err := store.SearchMemories(ctx, vector(1, 0), postgres.MemoryFilter{}, 10)
	if err != nil {
		t.Fatalf("SearchMemories: %v", err)
	}
	count := 0
	for _, r := range results {
		if r.Slug == "multi-chunk" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("multi-chunk appeared %d times in results, want exactly 1 despite having 3 matching chunks", count)
	}
}

func TestSearchMemoriesToleratesAMemoryWithNoChunks(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	// no Chunks set at all — the pre-backfill / not-yet-chunked case.
	_, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "no-chunks", Description: "d", Body: "b",
		Embedding: vector(1, 0), EmbeddingModel: "embeddinggemma",
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	results, err := store.SearchMemories(ctx, vector(1, 0), postgres.MemoryFilter{}, 10)
	if err != nil {
		t.Fatalf("SearchMemories: %v", err)
	}
	if len(results) != 1 || results[0].Slug != "no-chunks" {
		t.Errorf("results = %+v, want the chunkless memory found via its description score", results)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test -tags integration ./internal/store/postgres/... -run 'TestSearchMemoriesFindsMatchViaBodyChunkNotJustDescription|TestSearchMemoriesDeduplicatesMultipleChunkHitsToOneRowPerMemory|TestSearchMemoriesToleratesAMemoryWithNoChunks' -v`
Expected: FAIL — `TestSearchMemoriesFindsMatchViaBodyChunkNotJustDescription` fails because today's `SearchMemories` only looks at `memories.embedding`, so `long-memory` (description embedding pointing away from the query) won't be found via its chunk at all.

- [ ] **Step 3: Rewrite `SearchMemories`**

```go
// internal/store/postgres/search.go
package postgres

import (
	"context"
	"fmt"
	"sort"

	pgvector "github.com/pgvector/pgvector-go"

	"github.com/lonhutt/recall/internal/models"
)

type ScoredMemory struct {
	models.Memory
	Score float64
}

// SearchMemories ranks memories by cosine similarity to embedding, most
// similar first, narrowed by f. Callers should set f.EmbeddingModel; the
// cosine distance between vectors from two different models doesn't mean
// anything.
//
// Ranking is two-phase: a memory's description embedding (memories.embedding)
// is the primary signal, since it's always small and reflects the whole
// memory; a memory's body-chunk embeddings (memory_chunks.embedding) are a
// secondary signal that surfaces a long memory when a query only matches
// content deep in its body. A memory's final score is the better of the
// two, so one long memory's several chunks can't crowd out other memories
// by occupying multiple slots in the ranking.
func (s *Store) SearchMemories(ctx context.Context, embedding []float32, f MemoryFilter, limit int) ([]ScoredMemory, error) {
	vec := pgvector.NewVector(embedding)

	descWhere, descFilterArgs := buildFilter(1, f)
	descArgs := append([]any{vec}, descFilterArgs...)
	descArgs = append(descArgs, limit)
	descQuery := fmt.Sprintf(`
		SELECT id, 1 - (embedding <=> $1) AS score
		FROM memories
		%s
		ORDER BY embedding <=> $1
		LIMIT $%d
	`, descWhere, len(descArgs))

	descScores, err := s.scoreByID(ctx, descQuery, descArgs)
	if err != nil {
		return nil, fmt.Errorf("searching description embeddings: %w", err)
	}

	chunkWhere, chunkFilterArgs := buildFilter(1, f)
	chunkArgs := append([]any{vec}, chunkFilterArgs...)
	chunkArgs = append(chunkArgs, limit)
	chunkQuery := fmt.Sprintf(`
		SELECT memory_id, score FROM (
			SELECT DISTINCT ON (mc.memory_id) mc.memory_id, 1 - (mc.embedding <=> $1) AS score
			FROM memory_chunks mc
			JOIN memories m ON m.id = mc.memory_id
			%s
			ORDER BY mc.memory_id, mc.embedding <=> $1
		) best
		ORDER BY score DESC
		LIMIT $%d
	`, chunkWhere, len(chunkArgs))

	chunkScores, err := s.scoreByID(ctx, chunkQuery, chunkArgs)
	if err != nil {
		return nil, fmt.Errorf("searching body chunk embeddings: %w", err)
	}

	best := make(map[string]float64, len(descScores)+len(chunkScores))
	for id, score := range descScores {
		best[id] = score
	}
	for id, score := range chunkScores {
		if score > best[id] {
			best[id] = score
		}
	}
	if len(best) == 0 {
		return nil, nil
	}

	ids := make([]string, 0, len(best))
	for id := range best {
		ids = append(ids, id)
	}
	rows, err := s.pool.Query(ctx, `SELECT `+memoryColumns+` FROM memories WHERE id = ANY($1::uuid[])`, ids)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []ScoredMemory
	for rows.Next() {
		m, err := scanMemory(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, ScoredMemory{Memory: *m, Score: best[m.ID]})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

// scoreByID runs query (which must select exactly (id uuid, score float8)
// per row) and returns the results as a map.
func (s *Store) scoreByID(ctx context.Context, query string, args []any) (map[string]float64, error) {
	rows, err := s.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	scores := map[string]float64{}
	for rows.Next() {
		var id string
		var score float64
		if err := rows.Scan(&id, &score); err != nil {
			return nil, err
		}
		scores[id] = score
	}
	return scores, rows.Err()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test -tags integration ./internal/store/postgres/... -v`
Expected: PASS for all postgres integration tests, including the 3 new search tests and the pre-existing `TestSearchMemoriesOrdersByCosineSimilarity` (which creates memories with no `Chunks`, exercising the "chunk score map empty, description score wins" path).

- [ ] **Step 5: Commit**

```bash
git add internal/store/postgres/search.go internal/store/postgres/search_test.go
git commit -m "rank search results by best of description or body-chunk match"
```

---

## Task 5: Wire chunking into `save_memory`/`update_memory`

**Files:**
- Create: `internal/tools/embed.go`
- Modify: `internal/tools/deps.go`
- Modify: `internal/tools/save_memory.go`
- Modify: `internal/tools/update_memory.go`
- Modify: `internal/tools/save_memory_test.go`
- Modify: `internal/tools/update_memory_test.go`
- Modify: `internal/tools/e2e_test.go`
- Modify: `cmd/recall/main.go`

**Interfaces:**
- Consumes: `chunking.Chunk` (Task 1), `Embedder.CountTokens` (Task 2), `postgres.NewMemory.Chunks` / `postgres.MemoryPatch.Chunks` (Task 3).
- Produces: `(*Tools).embedDescription(ctx, description string) ([]float32, error)` and `(*Tools).chunkAndEmbedBody(ctx, body string) ([]postgres.MemoryChunk, error)` — both used again by Task 6's backfill command.

- [ ] **Step 1: Write the failing tests**

```go
// internal/tools/save_memory_test.go — replace TestSaveMemorySucceeds and add one test.
// (Other tests in this file — TestSaveMemoryRejectsUnknownType,
// TestSaveMemoryRejectsEmptyRequiredFields — fail validation before any
// embedding call and need no changes.)

func TestSaveMemorySucceeds(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1, 2, 3}}
	tools := &Tools{Store: store, Embedder: embedder, MaxTokensPerChunk: 2048}

	_, result, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "user", Slug: "likes-go", Description: "likes Go", Body: "prefers Go for backend work",
	})
	if err != nil {
		t.Fatalf("SaveMemory returned error: %v", err)
	}
	if result.Memory.Slug != "likes-go" {
		t.Errorf("Memory.Slug = %q, want likes-go", result.Memory.Slug)
	}
	if embedder.calls != 2 {
		t.Fatalf("embedder called %d times, want 2 (description, then body chunks): %+v", embedder.calls, embedder.allCalls)
	}
	if embedder.allCalls[0].texts[0] != "likes Go" {
		t.Errorf("first embed call = %v, want the description alone", embedder.allCalls[0].texts)
	}
	if embedder.allCalls[1].texts[0] != "prefers Go for backend work" {
		t.Errorf("second embed call = %v, want the body chunk", embedder.allCalls[1].texts)
	}
	if embedder.lastKind != embeddings.InputTypeDocument {
		t.Errorf("embedder kind = %q, want document", embedder.lastKind)
	}
}

func TestSaveMemoryChunksAnOversizedBody(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1, 2, 3}}
	tools := &Tools{Store: store, Embedder: embedder, MaxTokensPerChunk: 5}

	longBody := "one two three four five six seven eight nine ten eleven twelve"
	_, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "user", Slug: "long-memory", Description: "a long one", Body: longBody,
	})
	if err != nil {
		t.Fatalf("SaveMemory: %v", err)
	}
	if len(store.lastChunks) < 2 {
		t.Fatalf("Store got %d chunks, want at least 2 for a body over the 5-token budget", len(store.lastChunks))
	}
	for i, c := range store.lastChunks {
		if c.ChunkIndex != i {
			t.Errorf("chunk[%d].ChunkIndex = %d, want %d", i, c.ChunkIndex, i)
		}
		if len(c.Embedding) == 0 {
			t.Errorf("chunk[%d] has no embedding", i)
		}
	}
}
```

Also update the two other save_memory tests that reach the embed/chunk path (`TestSaveMemoryDuplicateSlugReturnsFriendlyError`, `TestSaveMemoryWrapsEmbeddingFailure`) to set `MaxTokensPerChunk: 2048` in their `Tools{}` literals:

```go
func TestSaveMemoryDuplicateSlugReturnsFriendlyError(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	args := SaveMemoryArgs{Type: "user", Slug: "dup", Description: "d", Body: "b"}

	if _, _, err := tools.SaveMemory(context.Background(), nil, args); err != nil {
		t.Fatalf("first SaveMemory: %v", err)
	}

	_, _, err := tools.SaveMemory(context.Background(), nil, args)
	if err == nil || !strings.Contains(err.Error(), "dup") || !strings.Contains(err.Error(), "update_memory") {
		t.Errorf("error = %v, want a message naming the slug and suggesting update_memory", err)
	}
}
```

`TestSaveMemoryWrapsEmbeddingFailure` needs no `MaxTokensPerChunk` change — `embedDescription` fails on the first `Embed` call, before `chunkAndEmbedBody` (and therefore `MaxTokensPerChunk`) is ever reached.

```go
// internal/tools/update_memory_test.go — replace TestUpdateMemoryReembedsOnBodyOrDescriptionChange
// with two more precise tests, and add MaxTokensPerChunk to every Tools{}
// literal in this file that reaches the embed/chunk path.

func TestUpdateMemoryReembedsBodyAloneWhenOnlyBodyChanges(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: store, Embedder: embedder, MaxTokensPerChunk: 2048}
	seedMemory(t, store, embedder, tools)
	embedder.calls = 0

	newBody := "new body"
	_, _, err := tools.UpdateMemory(context.Background(), nil, UpdateMemoryArgs{Slug: "seeded", Body: &newBody})
	if err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}
	if embedder.calls != 1 {
		t.Errorf("embedder called %d times, want 1 (body chunk only; description unchanged)", embedder.calls)
	}
	if embedder.lastKind != embeddings.InputTypeDocument {
		t.Errorf("embedder kind = %q, want document", embedder.lastKind)
	}
	if len(embedder.lastTexts) != 1 || embedder.lastTexts[0] != "new body" {
		t.Errorf("embedder texts = %v, want just the new body, not merged with description", embedder.lastTexts)
	}
}

func TestUpdateMemoryReembedsDescriptionAloneWhenOnlyDescriptionChanges(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: store, Embedder: embedder, MaxTokensPerChunk: 2048}
	seedMemory(t, store, embedder, tools)
	embedder.calls = 0

	newDesc := "new desc"
	_, _, err := tools.UpdateMemory(context.Background(), nil, UpdateMemoryArgs{Slug: "seeded", Description: &newDesc})
	if err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}
	if embedder.calls != 1 {
		t.Errorf("embedder called %d times, want 1 (description only; body unchanged)", embedder.calls)
	}
	if len(embedder.lastTexts) != 1 || embedder.lastTexts[0] != "new desc" {
		t.Errorf("embedder texts = %v, want just the new description", embedder.lastTexts)
	}
}
```

`TestUpdateMemoryPatchesOnlyGivenFields` needs no change (doesn't inspect embedder calls). `TestUpdateMemorySkipsEmbeddingWhenNeitherDescriptionNorBodyChange` needs no change (asserts 0 calls, still true). `TestUpdateMemoryNotFoundReturnsFriendlyError` and `TestUpdateMemoryRejectsUnknownType` need no change (fail before any embed/chunk call). `TestUpdateMemoryWrapsEmbeddingFailure` needs `MaxTokensPerChunk: 2048` added to its `Tools{}` literal:

```go
func TestUpdateMemoryWrapsEmbeddingFailure(t *testing.T) {
	store := newFakeStore()
	seedEmbedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: store, Embedder: seedEmbedder, MaxTokensPerChunk: 2048}
	seedMemory(t, store, seedEmbedder, tools)

	embedFailure := errors.New("llama.cpp server returned 500")
	tools.Embedder = &fakeEmbedder{err: embedFailure}

	newBody := "b2"
	_, _, err := tools.UpdateMemory(context.Background(), nil, UpdateMemoryArgs{Slug: "seeded", Body: &newBody})
	if !errors.Is(err, embedFailure) {
		t.Errorf("error = %v, want it to wrap %v", err, embedFailure)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./internal/tools/... -v`
Expected: FAIL to build — `Tools.MaxTokensPerChunk` doesn't exist yet, `embedDescription`/`chunkAndEmbedBody` don't exist yet.

- [ ] **Step 3: Implement**

```go
// internal/tools/deps.go — add MaxTokensPerChunk to Tools
type Tools struct {
	Store             Store
	Embedder          Embedder
	EmbeddingModel    string
	MaxTokensPerChunk int
}
```

```go
// internal/tools/embed.go
package tools

import (
	"context"
	"fmt"

	"github.com/lonhutt/recall/internal/chunking"
	"github.com/lonhutt/recall/internal/embeddings"
	"github.com/lonhutt/recall/internal/store/postgres"
)

// embedDescription embeds description alone as a memory's primary search
// vector. A description is documented as a one-line summary, so unlike
// body it's never expected to need chunking.
func (t *Tools) embedDescription(ctx context.Context, description string) ([]float32, error) {
	vectors, err := t.Embedder.Embed(ctx, []string{description}, embeddings.InputTypeDocument)
	if err != nil {
		return nil, err
	}
	return vectors[0], nil
}

// chunkAndEmbedBody splits body into pieces that fit t.MaxTokensPerChunk
// and embeds each one, for the secondary chunk-level search layer. A body
// that already fits produces exactly one chunk.
func (t *Tools) chunkAndEmbedBody(ctx context.Context, body string) ([]postgres.MemoryChunk, error) {
	texts, err := chunking.Chunk(ctx, body, t.MaxTokensPerChunk, t.Embedder.CountTokens)
	if err != nil {
		return nil, fmt.Errorf("chunking body: %w", err)
	}
	vectors, err := t.Embedder.Embed(ctx, texts, embeddings.InputTypeDocument)
	if err != nil {
		return nil, err
	}
	chunks := make([]postgres.MemoryChunk, len(texts))
	for i, text := range texts {
		chunks[i] = postgres.MemoryChunk{ChunkIndex: i, Text: text, Embedding: vectors[i]}
	}
	return chunks, nil
}
```

```go
// internal/tools/save_memory.go — full replacement
package tools

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/lonhutt/recall/internal/models"
	"github.com/lonhutt/recall/internal/store/postgres"
)

type SaveMemoryArgs struct {
	Type        string   `json:"type" jsonschema:"memory type: user, feedback, project, or reference"`
	Slug        string   `json:"slug" jsonschema:"short unique url-safe identifier, e.g. go-error-wrapping"`
	Description string   `json:"description" jsonschema:"one-line summary of this memory"`
	Body        string   `json:"body" jsonschema:"full memory content; may use [[slug]] links to other memories"`
	Project     string   `json:"project,omitempty" jsonschema:"project this memory relates to, if any"`
	Agent       string   `json:"agent,omitempty" jsonschema:"agent/client saving this, e.g. claude-code"`
	Tags        []string `json:"tags,omitempty" jsonschema:"arbitrary labels for filtering"`
}

type SaveMemoryResult struct {
	Memory MemoryDTO `json:"memory"`
}

func (t *Tools) SaveMemory(ctx context.Context, _ *mcp.CallToolRequest, a SaveMemoryArgs) (*mcp.CallToolResult, SaveMemoryResult, error) {
	if !models.IsValidMemoryType(a.Type) {
		return nil, SaveMemoryResult{}, fmt.Errorf("unknown memory type %q; expected one of %v", a.Type, models.ValidMemoryTypes())
	}
	if strings.TrimSpace(a.Slug) == "" {
		return nil, SaveMemoryResult{}, fmt.Errorf("slug is required")
	}
	if strings.TrimSpace(a.Description) == "" {
		return nil, SaveMemoryResult{}, fmt.Errorf("description is required")
	}
	if strings.TrimSpace(a.Body) == "" {
		return nil, SaveMemoryResult{}, fmt.Errorf("body is required")
	}

	descVec, err := t.embedDescription(ctx, a.Description)
	if err != nil {
		return nil, SaveMemoryResult{}, fmt.Errorf("generating description embedding: %w", err)
	}
	chunks, err := t.chunkAndEmbedBody(ctx, a.Body)
	if err != nil {
		return nil, SaveMemoryResult{}, fmt.Errorf("generating body chunk embeddings: %w", err)
	}

	mem, err := t.Store.CreateMemory(ctx, postgres.NewMemory{
		Type:           a.Type,
		Slug:           a.Slug,
		Description:    a.Description,
		Body:           a.Body,
		Project:        a.Project,
		Agent:          a.Agent,
		Tags:           a.Tags,
		Embedding:      descVec,
		EmbeddingModel: t.EmbeddingModel,
		Chunks:         chunks,
	})
	if err != nil {
		if errors.Is(err, models.ErrDuplicateSlug) {
			return nil, SaveMemoryResult{}, fmt.Errorf("a memory with slug %q already exists; use update_memory to modify it", a.Slug)
		}
		return nil, SaveMemoryResult{}, err
	}

	return nil, SaveMemoryResult{Memory: newMemoryDTO(mem)}, nil
}
```

```go
// internal/tools/update_memory.go — full replacement
package tools

import (
	"context"
	"errors"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/lonhutt/recall/internal/models"
	"github.com/lonhutt/recall/internal/store/postgres"
)

// UpdateMemoryArgs gives update_memory PATCH semantics: a nil field is left
// unchanged. Slug identifies the memory and is immutable.
type UpdateMemoryArgs struct {
	Slug        string    `json:"slug" jsonschema:"slug of the memory to update"`
	Type        *string   `json:"type,omitempty" jsonschema:"new memory type: user, feedback, project, or reference"`
	Description *string   `json:"description,omitempty"`
	Body        *string   `json:"body,omitempty"`
	Project     *string   `json:"project,omitempty"`
	Agent       *string   `json:"agent,omitempty"`
	Tags        *[]string `json:"tags,omitempty"`
}

type UpdateMemoryResult struct {
	Memory MemoryDTO `json:"memory"`
}

func (t *Tools) UpdateMemory(ctx context.Context, _ *mcp.CallToolRequest, a UpdateMemoryArgs) (*mcp.CallToolResult, UpdateMemoryResult, error) {
	if a.Type != nil && !models.IsValidMemoryType(*a.Type) {
		return nil, UpdateMemoryResult{}, fmt.Errorf("unknown memory type %q; expected one of %v", *a.Type, models.ValidMemoryTypes())
	}

	if _, err := t.Store.GetMemoryBySlug(ctx, a.Slug); err != nil {
		if errors.Is(err, models.ErrNotFound) {
			return nil, UpdateMemoryResult{}, fmt.Errorf("no memory found with slug %q", a.Slug)
		}
		return nil, UpdateMemoryResult{}, err
	}

	patch := postgres.MemoryPatch{
		Type:        a.Type,
		Description: a.Description,
		Body:        a.Body,
		Project:     a.Project,
		Agent:       a.Agent,
		Tags:        a.Tags,
	}

	// Description and body now embed independently: each is only
	// re-embedded when the caller actually changed it, not both whenever
	// either changes.
	if a.Description != nil || a.Body != nil {
		patch.EmbeddingModel = &t.EmbeddingModel
	}
	if a.Description != nil {
		vec, err := t.embedDescription(ctx, *a.Description)
		if err != nil {
			return nil, UpdateMemoryResult{}, fmt.Errorf("generating description embedding: %w", err)
		}
		patch.Embedding = vec
	}
	if a.Body != nil {
		chunks, err := t.chunkAndEmbedBody(ctx, *a.Body)
		if err != nil {
			return nil, UpdateMemoryResult{}, fmt.Errorf("generating body chunk embeddings: %w", err)
		}
		patch.Chunks = &chunks
	}

	updated, err := t.Store.UpdateMemory(ctx, a.Slug, patch)
	if err != nil {
		return nil, UpdateMemoryResult{}, err
	}
	return nil, UpdateMemoryResult{Memory: newMemoryDTO(updated)}, nil
}
```

```go
// cmd/recall/main.go — thread cfg.EmbeddingMaxTokens through to Tools
	t := &tools.Tools{
		Store:             store,
		Embedder:          embedder,
		EmbeddingModel:    cfg.EmbeddingModel,
		MaxTokensPerChunk: cfg.EmbeddingMaxTokens,
	}
```

```go
// internal/tools/e2e_test.go — TestMCPServerExposesAllToolsEndToEnd currently
// builds its Tools{} without MaxTokensPerChunk, which defaults to 0; the new
// save_memory/update_memory code path calls chunking.Chunk(ctx, body, 0, ...),
// which errors ("maxTokens must be positive"). Set it explicitly:
	toolset := &Tools{Store: store, Embedder: embedder, EmbeddingModel: "embeddinggemma", MaxTokensPerChunk: 2048}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go build ./... && go vet ./... && go test ./...`
Expected: PASS across the whole non-integration suite, including `TestMCPServerExposesAllToolsEndToEnd` (which exercises `save_memory`/`update_memory`/`search_memory` through the real tool-registration path). Then confirm the integration suite (from Tasks 3 and 4) is still green, since `notes_test.go`'s existing tests and this task's changes both touch `CreateMemory`/`UpdateMemory` call sites indirectly:

Run: `go test -tags integration ./...`
Expected: PASS (or SKIP if Docker isn't reachable — check `docker ps` first).

- [ ] **Step 5: Commit**

```bash
git add internal/tools/embed.go internal/tools/deps.go internal/tools/save_memory.go internal/tools/update_memory.go internal/tools/save_memory_test.go internal/tools/update_memory_test.go internal/tools/e2e_test.go cmd/recall/main.go
git commit -m "embed description and body chunks independently in save/update_memory"
```

---

## Task 6: `recall backfill-chunks` command

**Files:**
- Modify: `internal/tools/embed.go`
- Modify: `cmd/recall/main.go`
- Test: `internal/tools/backfill_test.go`

**Interfaces:**
- Consumes: `(*Tools).embedDescription`, `(*Tools).chunkAndEmbedBody` (Task 5); `Store.ListMemories` (existing, unchanged).
- Produces: `(*Tools).BackfillMemoryEmbedding(ctx, slug, description, body string) error`; `recall backfill-chunks` CLI subcommand.

- [ ] **Step 1: Write the failing tests**

```go
// internal/tools/backfill_test.go
package tools

import (
	"context"
	"errors"
	"testing"
)

func TestBackfillMemoryEmbeddingRegeneratesDescriptionAndChunks(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1, 2, 3}}
	tools := &Tools{Store: store, Embedder: embedder, EmbeddingModel: "embeddinggemma", MaxTokensPerChunk: 2048}
	seedMemory(t, store, embedder, tools)

	if err := tools.BackfillMemoryEmbedding(context.Background(), "seeded", "old desc", "old body"); err != nil {
		t.Fatalf("BackfillMemoryEmbedding: %v", err)
	}
	if len(store.lastChunks) != 1 || store.lastChunks[0].Text != "old body" {
		t.Errorf("lastChunks = %v, want a single chunk with the body text", store.lastChunks)
	}
}

func TestBackfillMemoryEmbeddingWrapsFailure(t *testing.T) {
	tools := &Tools{Store: newFakeStore(), Embedder: &fakeEmbedder{err: errors.New("boom")}, MaxTokensPerChunk: 2048}
	if err := tools.BackfillMemoryEmbedding(context.Background(), "missing", "d", "b"); err == nil {
		t.Fatal("expected an error, got nil")
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./internal/tools/... -run TestBackfillMemoryEmbedding -v`
Expected: FAIL to build — `BackfillMemoryEmbedding` undefined.

- [ ] **Step 3: Implement**

```go
// internal/tools/embed.go — append
// BackfillMemoryEmbedding regenerates slug's description embedding and
// body chunks from scratch. It's for the backfill-chunks command, to
// migrate memories saved before chunked search existed; normal save/update
// flows never need to call it directly.
func (t *Tools) BackfillMemoryEmbedding(ctx context.Context, slug, description, body string) error {
	descVec, err := t.embedDescription(ctx, description)
	if err != nil {
		return fmt.Errorf("generating description embedding: %w", err)
	}
	chunks, err := t.chunkAndEmbedBody(ctx, body)
	if err != nil {
		return fmt.Errorf("generating body chunk embeddings: %w", err)
	}
	_, err = t.Store.UpdateMemory(ctx, slug, postgres.MemoryPatch{
		Embedding:      descVec,
		EmbeddingModel: &t.EmbeddingModel,
		Chunks:         &chunks,
	})
	return err
}
```

```go
// cmd/recall/main.go — add the subcommand to run()'s switch
	switch cmd {
	case "migrate":
		dbURL := os.Getenv("RECALL_DATABASE_URL")
		if dbURL == "" {
			return fmt.Errorf("missing required environment variable: RECALL_DATABASE_URL")
		}
		return postgres.RunMigrations(dbURL)
	case "backfill-chunks":
		cfg, err := config.Load()
		if err != nil {
			return err
		}
		return backfillChunks(cfg)
	case "serve":
		cfg, err := config.Load()
		if err != nil {
			return err
		}
		return serve(cfg)
	default:
		return fmt.Errorf("unknown command %q; expected \"serve\", \"migrate\", or \"backfill-chunks\"", cmd)
	}
```

```go
// cmd/recall/main.go — new function, alongside serve()
// backfillChunks re-embeds every existing memory's description and body
// chunks under the current chunked-search scheme. Safe to run more than
// once (each run fully replaces the previous chunk set) and safe to run
// while recall is serving traffic (each memory's update is its own
// transaction, same as a normal update_memory call).
func backfillChunks(cfg *config.Config) error {
	ctx := context.Background()

	store, err := postgres.New(ctx, cfg.DatabaseURL)
	if err != nil {
		return err
	}
	defer store.Close()

	embedder := llamacpp.New(cfg.EmbeddingBaseURL, cfg.EmbeddingModel, cfg.EmbeddingQueryPrefix, cfg.EmbeddingDocumentPrefix, cfg.EmbeddingMaxTokens)
	t := &tools.Tools{Store: store, Embedder: embedder, EmbeddingModel: cfg.EmbeddingModel, MaxTokensPerChunk: cfg.EmbeddingMaxTokens}

	const pageSize = 50
	offset := 0
	for {
		memories, err := store.ListMemories(ctx, postgres.MemoryFilter{}, pageSize, offset)
		if err != nil {
			return fmt.Errorf("listing memories: %w", err)
		}
		if len(memories) == 0 {
			break
		}
		for _, m := range memories {
			if err := t.BackfillMemoryEmbedding(ctx, m.Slug, m.Description, m.Body); err != nil {
				return fmt.Errorf("backfilling %q: %w", m.Slug, err)
			}
			slog.Info("backfilled memory chunks", "slug", m.Slug)
		}
		offset += len(memories)
	}
	return nil
}
```

No new imports are needed in `main.go` — `postgres`, `tools`, `llamacpp`, `config`, `slog`, `context`, `fmt` are all already imported.

- [ ] **Step 4: Run tests to verify they pass**

Run: `go build ./... && go vet ./... && go test ./...`
Expected: PASS across the whole suite.

- [ ] **Step 5: Commit**

```bash
git add internal/tools/embed.go internal/tools/backfill_test.go cmd/recall/main.go
git commit -m "add recall backfill-chunks command"
```

---

## Final verification (after all 6 tasks)

Run, in order:

1. `gofmt -l .` — expect no output (nothing unformatted).
2. `go vet ./...` — expect no output.
3. `go build ./...` — expect success.
4. `go test ./...` — expect all packages `ok`.
5. `docker ps` to confirm Docker is reachable, then `go test -tags integration ./...` — expect all packages `ok` (postgres integration suite covers Tasks 3 and 4's real-database behavior that the fake-based unit tests can't).
6. Manually run `go run ./cmd/recall migrate` against a scratch database (or rely on `RECALL_MIGRATE_ON_START=true`, the default) to confirm migration `000006` applies cleanly on top of the existing 5.

Do not run `backfill-chunks` against the live `recall` container's database as part of this verification — that's a deliberate, separate step the person running this plan should take (and confirm with the user first), not something to do automatically once the code compiles and tests pass.
