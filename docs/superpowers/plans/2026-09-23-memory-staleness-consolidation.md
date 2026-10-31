# Memory Staleness & Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** recall can mark a memory as superseded (dedicated tool) or as a project-scoped override of a global one (declared attribute); both are demoted and clearly marked in search rather than competing at face value with current facts; recency decays a memory's ranking weight by type; and `save_memory` blocks on a likely duplicate (bypassable) instead of silently accumulating one.

**Architecture:** Two new nullable FK columns on `memories` (`superseded_by`, `overrides`, both `ON DELETE SET NULL`) carry the two relationship types. The MCP-facing layer always speaks in slugs; the store layer resolves slug↔id at its boundary via a shared `idForSlug` helper and inline scalar subqueries in `memoryColumns`. `SearchMemories` gains a post-processing pass over its existing fused (vector + keyword RRF) result set: tier assignment (superseded/overridden-in-scope sinks below normal) and an exponential recency-decay multiplier by type on the fused `Score`, both computed after the existing row fetch, not inside it. Write-time duplicate detection is a new, simpler single-phase similarity query, independent of search ranking.

**Tech Stack:** Go 1.27.1, Postgres 17 + pgvector (`pgx/v5`), existing MCP tool patterns. No new dependencies (`math` for decay is stdlib).

**Spec:** Obsidian `recall/Specs/2026-09-23-memory-staleness-consolidation-design.md` — the plan below reproduces every value/decision from it inline, so no separate lookup is needed to execute a task, but the design rationale lives there.

## Global Constraints

- No new dependencies.
- The MCP-facing layer (tool args/results) always uses slugs, never raw UUIDs, for `superseded_by`/`overrides` — matches every other cross-reference in this codebase (`[[slug]]` links, `update_memory`'s `slug` param).
- `overrides` is specifically project-overrides-global: the declaring memory must be project-scoped, its target must be global. Validation error otherwise (including the case that would otherwise permit a cycle).
- `supersede_memory` is the only way to *set* `superseded_by` to a non-null value; `update_memory` can only clear it. Keeps a single entry point for creating that relationship.
- Decay half-lives (720 days for `user`/`reference`, 90 days for `project`/`feedback`) and the duplicate-detection similarity threshold (0.92) are deliberately hardcoded constants, not env vars — a corner cut; expose them later if real usage shows the defaults are wrong.
- `save_memory` now requires an explicit `project` or `global: true` — a caller omitting both gets a validation error, never a silent global save. This is a breaking change for any caller that previously omitted `project` (see Task 7 — the recall skill itself must supply one or the other on every call).
- Follow existing test conventions exactly: `fakeStore`/`fakeEmbedder` unit tests in `internal/tools`, `//go:build integration` + testcontainers in `internal/store/postgres`.

## Review Focus

- A memory with neither `superseded_by` nor an in-scope override must rank and behave exactly as it does today — no regression for the common case (Task 5).
- Deleting a memory that something else points to via `superseded_by`/`overrides` must clear the pointer (`ON DELETE SET NULL`), never leave a dangling reference or block the delete (Task 1, Task 3).
- An `overrides` value must be rejected if the declaring memory is global, the target doesn't exist, or the target is itself project-scoped — all three are real inputs an agent could pass by mistake (Task 4).
- A `search_memory` call with no `project` filter must never apply override-demotion (it's scope-conditional by definition) — only `superseded_by` demotion is scope-independent (Task 5).
- `save_memory`'s duplicate check must never block a legitimately distinct memory that happens to score high similarity without an escape hatch — `force: true` must always be honored (Task 6).
- After Task 5, rerun `docker compose run --rm -T recall eval < ~/.local/share/recall/eval/probes.json` and compare against the Results in Obsidian `recall/Plans/2026-10-31-hybrid-search-and-retrieval-eval.md` (hybrid: hits@1 31/34, hits@3 34/34, MRR 0.956). Decay reorders fused scores, so a drop on situation probes there means the multiplier is fighting RRF (Task 5).

---

## Task 1: Schema migration

**Files:**
- Create: `internal/store/postgres/migrations/000008_add_memory_relationships.up.sql`
- Create: `internal/store/postgres/migrations/000008_add_memory_relationships.down.sql`

**Interfaces:**
- Consumes: nothing (pure schema change).
- Produces: `memories.superseded_by uuid`, `memories.overrides uuid` — consumed by every later task.

- [ ] **Step 1: Write the migration**

```sql
-- internal/store/postgres/migrations/000008_add_memory_relationships.up.sql
ALTER TABLE memories ADD COLUMN superseded_by uuid REFERENCES memories(id) ON DELETE SET NULL;
ALTER TABLE memories ADD COLUMN overrides uuid REFERENCES memories(id) ON DELETE SET NULL;

CREATE INDEX memories_superseded_by_idx ON memories (superseded_by) WHERE superseded_by IS NOT NULL;
CREATE INDEX memories_overrides_idx ON memories (overrides) WHERE overrides IS NOT NULL;
```

```sql
-- internal/store/postgres/migrations/000008_add_memory_relationships.down.sql
ALTER TABLE memories DROP COLUMN superseded_by;
ALTER TABLE memories DROP COLUMN overrides;
```

(Dropping a column drops its index and FK automatically — no explicit `DROP INDEX` needed, matching migration `000006`'s down-migration style.)

- [ ] **Step 2: Verify the migration applies cleanly**

Run: `go build ./... && go test -tags integration ./internal/store/postgres/... -run TestCreateAndGetMemory -v` (from repo root)
Expected: PASS. This doesn't exercise the new columns yet (Task 2 does), but it proves migration `000008` applies without error on top of `000001`-`000007` and doesn't break the existing schema — the embedded migration set runs automatically when `newTestStore(t)` spins up its container.

- [ ] **Step 3: Commit**

```bash
git add internal/store/postgres/migrations/000008_add_memory_relationships.up.sql internal/store/postgres/migrations/000008_add_memory_relationships.down.sql
git commit -m "add superseded_by and overrides columns to memories"
```

---

## Task 2: Read path — `models.Memory`, `memoryColumns`/`scanMemory`, `idForSlug`

**Files:**
- Modify: `internal/models/memory.go`
- Modify: `internal/store/postgres/notes.go`
- Test: `internal/store/postgres/notes_test.go`

**Interfaces:**
- Consumes: Task 1's columns.
- Produces: `models.Memory.SupersededBy string` / `.Overrides string` (slug, `""` = unset); `idForSlug(ctx, q queryRower, slug string) (string, error)` — used by Tasks 3 and 4 to resolve a slug to the id a FK column needs.

- [ ] **Step 1: Write the failing test**

`idForSlug` and the relationship-set-then-read round trip both get their real coverage from real callers in Task 3 (`supersede_memory`) and Task 4 (`overrides`) — this codebase never exports test-only surface (no `*ForTest` methods exist anywhere in it), so Task 2's own test is scoped to exactly what Task 2 itself adds: the read path returning empty relationship fields by default, using `CreateMemory`'s existing, untouched behavior.

```go
// append to internal/store/postgres/notes_test.go
func TestCreateMemoryLeavesRelationshipFieldsEmptyByDefault(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	mem, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "plain", Description: "d", Body: "b", Embedding: vector(1, 0),
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}
	if mem.SupersededBy != "" || mem.Overrides != "" {
		t.Errorf("SupersededBy = %q, Overrides = %q, want both empty for a memory that doesn't declare either", mem.SupersededBy, mem.Overrides)
	}

	got, err := store.GetMemoryBySlug(ctx, "plain")
	if err != nil {
		t.Fatalf("GetMemoryBySlug: %v", err)
	}
	if got.SupersededBy != "" || got.Overrides != "" {
		t.Errorf("re-fetched SupersededBy = %q, Overrides = %q, want both empty", got.SupersededBy, got.Overrides)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test -tags integration ./internal/store/postgres/... -run TestCreateMemoryLeavesRelationshipFieldsEmptyByDefault -v`
Expected: FAIL to build — `models.Memory` has no `SupersededBy`/`Overrides` fields yet.

- [ ] **Step 3: Implement**

```go
// internal/models/memory.go — add two fields to Memory
type Memory struct {
	ID             string
	Type           string
	Slug           string
	Description    string
	Body           string
	Project        string
	Agent          string
	Tags           []string
	EmbeddingModel string
	SupersededBy   string
	Overrides      string
	CreatedAt      time.Time
	UpdatedAt      time.Time
}
```

```go
// internal/store/postgres/notes.go — extend memoryColumns and scanMemory

const memoryColumns = `id, type, slug, description, body, project, agent, tags, embedding_model, created_at, updated_at,
	(SELECT slug FROM memories sb WHERE sb.id = memories.superseded_by) AS superseded_by_slug,
	(SELECT slug FROM memories ov WHERE ov.id = memories.overrides) AS overrides_slug`

// scanMemory reads one row selected as memoryColumns; extra destinations are
// appended in order, for callers that select a trailing expression alongside
// the standard columns.
func scanMemory(row rowScanner, extra ...any) (*models.Memory, error) {
	var (
		m            models.Memory
		project      sql.NullString
		agent        sql.NullString
		supersededBy sql.NullString
		overrides    sql.NullString
	)
	dest := []any{&m.ID, &m.Type, &m.Slug, &m.Description, &m.Body, &project, &agent,
		&m.Tags, &m.EmbeddingModel, &m.CreatedAt, &m.UpdatedAt, &supersededBy, &overrides}
	err := row.Scan(append(dest, extra...)...)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, models.ErrNotFound
		}
		return nil, err
	}
	m.Project = project.String
	m.Agent = agent.String
	m.SupersededBy = supersededBy.String
	m.Overrides = overrides.String
	return &m, nil
}
```

```go
// internal/store/postgres/notes.go — add near the top, after scanMemory

// queryRower is satisfied by both *pgxpool.Pool and pgx.Tx, so idForSlug
// works inside or outside a transaction.
type queryRower interface {
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

// idForSlug resolves slug to the row's id, or models.ErrNotFound if no
// memory has that slug. Used wherever a FK column (superseded_by,
// overrides) needs to be written from the slug the MCP-facing layer works in.
func idForSlug(ctx context.Context, q queryRower, slug string) (string, error) {
	var id string
	err := q.QueryRow(ctx, `SELECT id FROM memories WHERE slug = $1`, slug).Scan(&id)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", models.ErrNotFound
	}
	return id, err
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test -tags integration ./internal/store/postgres/... -v`
Expected: PASS for all postgres integration tests, including the new one and every pre-existing one (the two new scalar subqueries in `memoryColumns` return null for every existing row, scanning cleanly to `""`).

Also: `go build ./... && go test ./...`
Expected: PASS (no other package touches `models.Memory`'s field list directly in a way that would break from two new fields).

- [ ] **Step 5: Commit**

```bash
git add internal/models/memory.go internal/store/postgres/notes.go internal/store/postgres/notes_test.go
git commit -m "read superseded_by/overrides as slugs on every memory fetch"
```

---

## Task 3: Supersession — `supersede_memory` tool + store write path

**Files:**
- Modify: `internal/store/postgres/notes.go`
- Create: `internal/tools/supersede_memory.go`
- Modify: `internal/tools/deps.go`
- Modify: `internal/tools/register.go`
- Test: `internal/store/postgres/notes_test.go`
- Test: `internal/tools/supersede_memory_test.go`

**Interfaces:**
- Consumes: `idForSlug`, `scanMemory`/`memoryColumns` (Task 2).
- Produces: `Store.SupersedeMemory(ctx, oldSlug, newSlug string) (*models.Memory, error)`; `MemoryPatch.ClearSupersededBy bool`; the `supersede_memory` MCP tool.

- [ ] **Step 1: Write the failing tests**

```go
// internal/store/postgres/notes_test.go — append
func TestSupersedeMemorySetsSupersededBy(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "old-fact", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(old-fact): %v", err)
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "new-fact", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(new-fact): %v", err)
	}

	updated, err := store.SupersedeMemory(ctx, "old-fact", "new-fact")
	if err != nil {
		t.Fatalf("SupersedeMemory: %v", err)
	}
	if updated.SupersededBy != "new-fact" {
		t.Errorf("SupersededBy = %q, want new-fact", updated.SupersededBy)
	}

	got, err := store.GetMemoryBySlug(ctx, "old-fact")
	if err != nil {
		t.Fatalf("GetMemoryBySlug: %v", err)
	}
	if got.SupersededBy != "new-fact" {
		t.Errorf("re-fetched SupersededBy = %q, want new-fact", got.SupersededBy)
	}
}

func TestSupersedeMemoryReturnsNotFoundForEitherMissingSlug(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "exists", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	if _, err := store.SupersedeMemory(ctx, "missing-old", "exists"); !errors.Is(err, models.ErrNotFound) {
		t.Errorf("old-slug-missing: err = %v, want models.ErrNotFound", err)
	}
	if _, err := store.SupersedeMemory(ctx, "exists", "missing-new"); !errors.Is(err, models.ErrNotFound) {
		t.Errorf("new-slug-missing: err = %v, want models.ErrNotFound", err)
	}
}

func TestUpdateMemoryClearsSupersededBy(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "old-fact-2", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(old-fact-2): %v", err)
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "new-fact-2", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(new-fact-2): %v", err)
	}
	if _, err := store.SupersedeMemory(ctx, "old-fact-2", "new-fact-2"); err != nil {
		t.Fatalf("SupersedeMemory: %v", err)
	}

	updated, err := store.UpdateMemory(ctx, "old-fact-2", postgres.MemoryPatch{ClearSupersededBy: true})
	if err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}
	if updated.SupersededBy != "" {
		t.Errorf("SupersededBy = %q after clearing, want empty", updated.SupersededBy)
	}
}

func TestUpdateMemoryWithoutClearSupersededByLeavesItAlone(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "old-fact-3", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(old-fact-3): %v", err)
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "new-fact-3", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(new-fact-3): %v", err)
	}
	if _, err := store.SupersedeMemory(ctx, "old-fact-3", "new-fact-3"); err != nil {
		t.Fatalf("SupersedeMemory: %v", err)
	}

	newProject := "recall"
	updated, err := store.UpdateMemory(ctx, "old-fact-3", postgres.MemoryPatch{Project: &newProject})
	if err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}
	if updated.SupersededBy != "new-fact-3" {
		t.Errorf("SupersededBy = %q, want it untouched by an unrelated patch", updated.SupersededBy)
	}
}
```

```go
// internal/tools/supersede_memory_test.go
package tools

import (
	"context"
	"errors"
	"testing"

	"github.com/lonhutt/recall/internal/models"
)

func TestSupersedeMemorySucceeds(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	seedMemory(t, store, &fakeEmbedder{vector: []float32{1}}, tools)
	if _, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "reference", Slug: "replacement", Description: "d", Body: "b", Global: true,
	}); err != nil {
		t.Fatalf("seeding replacement: %v", err)
	}

	_, result, err := tools.SupersedeMemory(context.Background(), nil, SupersedeMemoryArgs{OldSlug: "seeded", NewSlug: "replacement"})
	if err != nil {
		t.Fatalf("SupersedeMemory: %v", err)
	}
	if result.Memory.SupersededBy != "replacement" {
		t.Errorf("SupersededBy = %q, want replacement", result.Memory.SupersededBy)
	}
}

func TestSupersedeMemoryRejectsMissingOldSlug(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	if _, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "reference", Slug: "replacement", Description: "d", Body: "b", Global: true,
	}); err != nil {
		t.Fatalf("seeding: %v", err)
	}

	_, _, err := tools.SupersedeMemory(context.Background(), nil, SupersedeMemoryArgs{OldSlug: "missing", NewSlug: "replacement"})
	if !errors.Is(err, models.ErrNotFound) {
		t.Errorf("err = %v, want it to wrap models.ErrNotFound", err)
	}
}

func TestSupersedeMemoryRejectsMissingNewSlug(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	seedMemory(t, store, &fakeEmbedder{vector: []float32{1}}, tools)

	_, _, err := tools.SupersedeMemory(context.Background(), nil, SupersedeMemoryArgs{OldSlug: "seeded", NewSlug: "missing"})
	if !errors.Is(err, models.ErrNotFound) {
		t.Errorf("err = %v, want it to wrap models.ErrNotFound", err)
	}
}

func TestSupersedeMemoryRejectsSelfSupersession(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	seedMemory(t, store, &fakeEmbedder{vector: []float32{1}}, tools)

	_, _, err := tools.SupersedeMemory(context.Background(), nil, SupersedeMemoryArgs{OldSlug: "seeded", NewSlug: "seeded"})
	if err == nil {
		t.Fatal("expected an error for old_slug == new_slug, got nil")
	}
}
```

The `fakeStore` doesn't have a `SupersedeMemory` method yet — this task adds it (see Step 3's `fakes_test.go` change).

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test -tags integration ./internal/store/postgres/... -run 'TestSupersedeMemory|TestUpdateMemoryClearsSupersededBy|TestUpdateMemoryWithoutClearSupersededByLeavesItAlone' -v` and `go test ./internal/tools/... -run TestSupersedeMemory -v`
Expected: FAIL to build — `Store.SupersedeMemory`, `MemoryPatch.ClearSupersededBy`, `SupersedeMemoryArgs`, `Tools.SupersedeMemory` don't exist yet.

- [ ] **Step 3: Implement**

```go
// internal/store/postgres/notes.go — add ClearSupersededBy to MemoryPatch
type MemoryPatch struct {
	Type              *string
	Description       *string
	Body              *string
	Project           *string
	Agent             *string
	Tags              *[]string
	Embedding         []float32
	EmbeddingModel    *string
	Chunks            *[]MemoryChunk
	ClearSupersededBy bool
	Overrides         *string // added here now; Task 4 is what actually sets it from the tools layer
}
```

```go
// internal/store/postgres/notes.go — in UpdateMemory, add handling for
// ClearSupersededBy and Overrides alongside the existing p.Chunks handling.
// This requires the transaction (tx) for the Overrides slug lookup, so
// place both blocks where p.Chunks is currently handled, before tx.Commit:

	if p.ClearSupersededBy {
		sets = append(sets, "superseded_by = NULL")
	}
```

That line goes in the `sets`-building block (alongside the existing `if p.Type != nil { set(...) }` chain), *before* `args = append(args, slug)` — since `sets` is built as a plain string with no placeholder for this one (no arg needed, `NULL` is a literal). Full updated `UpdateMemory` for this task (the `Overrides` handling shown here as a placeholder that panics if ever hit — Task 4 replaces it with the real implementation; this keeps Task 3's diff buildable without jumping ahead into Task 4's validation logic):

```go
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
	if p.ClearSupersededBy {
		sets = append(sets, "superseded_by = NULL")
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

	if p.Overrides != nil {
		if err := applyOverridesPatch(ctx, tx, slug, *p.Overrides); err != nil {
			return nil, err
		}
	}

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

// applyOverridesPatch is a placeholder for Task 3 — Task 4 replaces this
// with the real slug-resolution + UPDATE. For now it's unreachable because
// no caller sets MemoryPatch.Overrides until Task 4 exists, but it must
// compile.
func applyOverridesPatch(ctx context.Context, tx pgx.Tx, slug, overrides string) error {
	return fmt.Errorf("overrides patch not implemented until Task 4")
}
```

```go
// internal/store/postgres/notes.go — new method, add after UpdateMemory

// SupersedeMemory marks oldSlug as superseded by newSlug. Both must already
// exist; errors name whichever is missing. No content on either memory
// changes — this only records the relationship.
func (s *Store) SupersedeMemory(ctx context.Context, oldSlug, newSlug string) (*models.Memory, error) {
	newID, err := idForSlug(ctx, s.pool, newSlug)
	if err != nil {
		return nil, err
	}
	row := s.pool.QueryRow(ctx, `UPDATE memories SET superseded_by = $1, updated_at = now() WHERE slug = $2 RETURNING `+memoryColumns,
		newID, oldSlug)
	return scanMemory(row)
}
```

```go
// internal/tools/supersede_memory.go
package tools

import (
	"context"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type SupersedeMemoryArgs struct {
	OldSlug string `json:"old_slug" jsonschema:"slug of the memory that's no longer current"`
	NewSlug string `json:"new_slug" jsonschema:"slug of the memory that replaces it"`
}

type SupersedeMemoryResult struct {
	Memory MemoryDTO `json:"memory"`
}

// SupersedeMemory marks old_slug as superseded by new_slug — search ranks
// a superseded memory below every current one and marks it with
// superseded_by, rather than competing with current facts at face value.
// Neither memory's content changes. Reversible via update_memory's
// clear_superseded_by.
func (t *Tools) SupersedeMemory(ctx context.Context, _ *mcp.CallToolRequest, a SupersedeMemoryArgs) (*mcp.CallToolResult, SupersedeMemoryResult, error) {
	if a.OldSlug == a.NewSlug {
		return nil, SupersedeMemoryResult{}, fmt.Errorf("old_slug and new_slug must be different (got %q for both)", a.OldSlug)
	}
	updated, err := t.Store.SupersedeMemory(ctx, a.OldSlug, a.NewSlug)
	if err != nil {
		return nil, SupersedeMemoryResult{}, err
	}
	return nil, SupersedeMemoryResult{Memory: newMemoryDTO(updated)}, nil
}
```

```go
// internal/tools/deps.go — add SupersedeMemory to the Store interface
type Store interface {
	CreateMemory(ctx context.Context, m postgres.NewMemory) (*models.Memory, error)
	UpdateMemory(ctx context.Context, slug string, p postgres.MemoryPatch) (*models.Memory, error)
	GetMemoryBySlug(ctx context.Context, slug string) (*models.Memory, error)
	DeleteMemory(ctx context.Context, slug string) error
	ListMemories(ctx context.Context, f postgres.MemoryFilter, limit, offset int) ([]models.Memory, error)
	SearchMemories(ctx context.Context, embedding []float32, query string, f postgres.MemoryFilter, limit int) ([]postgres.ScoredMemory, error)
	CreateLogEvent(ctx context.Context, e postgres.NewLogEvent) (*models.LogEvent, error)
	ListLogEvents(ctx context.Context, f postgres.EventFilter, limit, offset int) ([]models.LogEvent, error)
	SupersedeMemory(ctx context.Context, oldSlug, newSlug string) (*models.Memory, error)
}
```

```go
// internal/tools/register.go — add to RegisterAll
	mcp.AddTool(s, &mcp.Tool{
		Name:        "supersede_memory",
		Description: "Mark a memory as superseded by another — search ranks it below current facts and flags it, rather than treating it as still true.",
	}, t.SupersedeMemory)
```

```go
// internal/tools/fakes_test.go — add to fakeStore
func (f *fakeStore) SupersedeMemory(_ context.Context, oldSlug, newSlug string) (*models.Memory, error) {
	if _, ok := f.bySlug[oldSlug]; !ok {
		return nil, models.ErrNotFound
	}
	if _, ok := f.bySlug[newSlug]; !ok {
		return nil, models.ErrNotFound
	}
	mem := f.bySlug[oldSlug]
	mem.SupersededBy = newSlug
	f.bySlug[oldSlug] = mem
	return &mem, nil
}
```

Also update `newMemoryDTO`/`MemoryDTO` (Task 2 added the model field; this task is the first to actually populate it in output) — this belongs here since `SupersedeMemory`'s result is the first place `SupersededBy` needs to appear over the wire:

```go
// internal/tools/dto.go — add to MemoryDTO
type MemoryDTO struct {
	ID           string   `json:"id"`
	Type         string   `json:"type"`
	Slug         string   `json:"slug"`
	Description  string   `json:"description"`
	Body         string   `json:"body"`
	Project      string   `json:"project,omitempty"`
	Agent        string   `json:"agent,omitempty"`
	Tags         []string `json:"tags,omitempty"`
	SupersededBy string   `json:"superseded_by,omitempty"`
	Overrides    string   `json:"overrides,omitempty"`
	CreatedAt    string   `json:"created_at"`
	UpdatedAt    string   `json:"updated_at"`
}

func newMemoryDTO(m *models.Memory) MemoryDTO {
	return MemoryDTO{
		ID:           m.ID,
		Type:         m.Type,
		Slug:         m.Slug,
		Description:  m.Description,
		Body:         m.Body,
		Project:      m.Project,
		Agent:        m.Agent,
		Tags:         m.Tags,
		SupersededBy: m.SupersededBy,
		Overrides:    m.Overrides,
		CreatedAt:    m.CreatedAt.Format(time.RFC3339),
		UpdatedAt:    m.UpdatedAt.Format(time.RFC3339),
	}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test -tags integration ./internal/store/postgres/... -v` and `go build ./... && go test ./...`
Expected: PASS everywhere, including the placeholder `applyOverridesPatch` (unreachable — nothing sets `MemoryPatch.Overrides` until Task 4, so it compiles but is never actually called by any test in this task).

- [ ] **Step 5: Commit**

```bash
git add internal/store/postgres/notes.go internal/store/postgres/notes_test.go internal/tools/supersede_memory.go internal/tools/supersede_memory_test.go internal/tools/deps.go internal/tools/register.go internal/tools/fakes_test.go internal/tools/dto.go
git commit -m "add supersede_memory tool"
```

---

## Task 4: Override — `overrides` on `save_memory`/`update_memory`, plus the explicit-scope requirement

**Files:**
- Modify: `internal/store/postgres/notes.go`
- Modify: `internal/tools/save_memory.go`
- Modify: `internal/tools/update_memory.go`
- Modify: `internal/tools/helpers.go`
- Test: `internal/store/postgres/notes_test.go`
- Test: `internal/tools/save_memory_test.go`
- Test: `internal/tools/update_memory_test.go`

**Interfaces:**
- Consumes: `idForSlug` (Task 2), `MemoryPatch.Overrides` field (declared in Task 3, implemented here).
- Produces: `(*Tools).validateOverrides(ctx, project, overrides string) error` — shared by both handlers; `NewMemory.Overrides string`.

- [ ] **Step 1: Write the failing tests**

```go
// internal/store/postgres/notes_test.go — append
func TestCreateMemoryWithOverridesResolvesToId(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "global-pref", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(global-pref): %v", err)
	}

	newProject := "recall"
	mem, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "project", Slug: "project-pref", Description: "d", Body: "b", Embedding: vector(1, 0),
		Project: newProject, Overrides: "global-pref",
	})
	if err != nil {
		t.Fatalf("CreateMemory(project-pref): %v", err)
	}
	if mem.Overrides != "global-pref" {
		t.Errorf("Overrides = %q, want global-pref", mem.Overrides)
	}
}

func TestUpdateMemorySetsAndClearsOverrides(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "global-pref-2", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(global-pref-2): %v", err)
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "project", Slug: "project-pref-2", Description: "d", Body: "b", Embedding: vector(1, 0), Project: "recall",
	}); err != nil {
		t.Fatalf("CreateMemory(project-pref-2): %v", err)
	}

	overrides := "global-pref-2"
	updated, err := store.UpdateMemory(ctx, "project-pref-2", postgres.MemoryPatch{Overrides: &overrides})
	if err != nil {
		t.Fatalf("UpdateMemory(set): %v", err)
	}
	if updated.Overrides != "global-pref-2" {
		t.Errorf("Overrides = %q after set, want global-pref-2", updated.Overrides)
	}

	empty := ""
	cleared, err := store.UpdateMemory(ctx, "project-pref-2", postgres.MemoryPatch{Overrides: &empty})
	if err != nil {
		t.Fatalf("UpdateMemory(clear): %v", err)
	}
	if cleared.Overrides != "" {
		t.Errorf("Overrides = %q after clear, want empty", cleared.Overrides)
	}
}

func TestDeletingAnOverriddenOrSupersedingMemoryClearsThePointer(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "global-pref-3", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(global-pref-3): %v", err)
	}
	overrides := "global-pref-3"
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "project", Slug: "project-pref-3", Description: "d", Body: "b", Embedding: vector(1, 0),
		Project: "recall", Overrides: overrides,
	}); err != nil {
		t.Fatalf("CreateMemory(project-pref-3): %v", err)
	}

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "old-fact-4", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(old-fact-4): %v", err)
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "new-fact-4", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(new-fact-4): %v", err)
	}
	if _, err := store.SupersedeMemory(ctx, "old-fact-4", "new-fact-4"); err != nil {
		t.Fatalf("SupersedeMemory: %v", err)
	}

	// Delete the target of each relationship — the FK's ON DELETE SET NULL
	// (Task 1) must clear the referencing row's pointer, not error and not
	// leave it dangling.
	if err := store.DeleteMemory(ctx, "global-pref-3"); err != nil {
		t.Fatalf("DeleteMemory(global-pref-3): %v", err)
	}
	if err := store.DeleteMemory(ctx, "new-fact-4"); err != nil {
		t.Fatalf("DeleteMemory(new-fact-4): %v", err)
	}

	gotOverride, err := store.GetMemoryBySlug(ctx, "project-pref-3")
	if err != nil {
		t.Fatalf("GetMemoryBySlug(project-pref-3): %v", err)
	}
	if gotOverride.Overrides != "" {
		t.Errorf("Overrides = %q after its target was deleted, want empty (ON DELETE SET NULL)", gotOverride.Overrides)
	}

	gotSuperseded, err := store.GetMemoryBySlug(ctx, "old-fact-4")
	if err != nil {
		t.Fatalf("GetMemoryBySlug(old-fact-4): %v", err)
	}
	if gotSuperseded.SupersededBy != "" {
		t.Errorf("SupersededBy = %q after its target was deleted, want empty (ON DELETE SET NULL)", gotSuperseded.SupersededBy)
	}
}
```

```go
// internal/tools/save_memory_test.go — append
func TestSaveMemoryRequiresProjectOrExplicitGlobal(t *testing.T) {
	tools := &Tools{Store: newFakeStore(), Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}

	_, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "user", Slug: "s", Description: "d", Body: "b",
	})
	if err == nil {
		t.Fatal("expected an error when project is omitted and global is not set, got nil")
	}
}

func TestSaveMemoryAllowsExplicitGlobal(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}

	_, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "user", Slug: "s", Description: "d", Body: "b", Global: true,
	})
	if err != nil {
		t.Fatalf("SaveMemory with Global: true: %v", err)
	}
}

func TestSaveMemoryWithOverridesRequiresProjectScope(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	if _, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "reference", Slug: "global-thing", Description: "d", Body: "b", Global: true,
	}); err != nil {
		t.Fatalf("seeding global-thing: %v", err)
	}

	_, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "reference", Slug: "another-global", Description: "d", Body: "b", Global: true, Overrides: "global-thing",
	})
	if err == nil {
		t.Fatal("expected an error: a global memory can't declare overrides, got nil")
	}
}

func TestSaveMemoryWithOverridesRejectsAProjectScopedTarget(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	if _, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "project", Slug: "project-thing", Description: "d", Body: "b", Project: "recall",
	}); err != nil {
		t.Fatalf("seeding project-thing: %v", err)
	}

	_, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "project", Slug: "other-project-thing", Description: "d", Body: "b", Project: "recall", Overrides: "project-thing",
	})
	if err == nil {
		t.Fatal("expected an error: overrides target must be global, got nil")
	}
}

func TestSaveMemoryWithOverridesRejectsAMissingTarget(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}

	_, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "project", Slug: "project-thing-2", Description: "d", Body: "b", Project: "recall", Overrides: "no-such-slug",
	})
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
}

func TestSaveMemoryWithOverridesSucceedsForAValidProjectOverGlobal(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	if _, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "reference", Slug: "global-tabs", Description: "prefers tabs", Body: "b", Global: true,
	}); err != nil {
		t.Fatalf("seeding global-tabs: %v", err)
	}

	_, result, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "project", Slug: "recall-spaces", Description: "uses spaces in this repo", Body: "b",
		Project: "recall", Overrides: "global-tabs",
	})
	if err != nil {
		t.Fatalf("SaveMemory: %v", err)
	}
	if result.Memory.Overrides != "global-tabs" {
		t.Errorf("Overrides = %q, want global-tabs", result.Memory.Overrides)
	}
}
```

Note: every OTHER existing `save_memory_test.go`/`update_memory_test.go`/`e2e_test.go`/`delete_memory_test.go`/`list_memories_test.go` test that calls `SaveMemory` needs `Global: true` (or a non-empty `Project`) added to its `SaveMemoryArgs` literal now that the project-or-global requirement is enforced — otherwise they'll all start failing with the new validation error. Add `Global: true` to every existing `SaveMemoryArgs{}` literal across the `internal/tools` test files that doesn't already set `Project`. This includes `seedMemory` in `update_memory_test.go` (add `Global: true` to its `SaveMemoryArgs`) and the `e2e_test.go` call to `save_memory`.

```go
// internal/tools/update_memory_test.go — append
func TestUpdateMemorySetsOverrides(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: store, Embedder: embedder, MaxTokensPerChunk: 2048}
	if _, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "reference", Slug: "global-x", Description: "d", Body: "b", Global: true,
	}); err != nil {
		t.Fatalf("seeding global-x: %v", err)
	}
	seedMemory(t, store, embedder, tools) // "seeded", now needs Global: true internally — see note above

	overrides := "global-x"
	_, result, err := tools.UpdateMemory(context.Background(), nil, UpdateMemoryArgs{Slug: "seeded", Overrides: &overrides})
	if err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}
	if result.Memory.Overrides != "global-x" {
		t.Errorf("Overrides = %q, want global-x", result.Memory.Overrides)
	}
}

func TestUpdateMemoryRejectsOverridesWhenTargetIsNotGlobal(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: store, Embedder: embedder, MaxTokensPerChunk: 2048}
	seedMemory(t, store, embedder, tools)
	if _, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "project", Slug: "also-project-scoped", Description: "d", Body: "b", Project: "recall",
	}); err != nil {
		t.Fatalf("seeding also-project-scoped: %v", err)
	}

	overrides := "also-project-scoped"
	_, _, err := tools.UpdateMemory(context.Background(), nil, UpdateMemoryArgs{Slug: "seeded", Overrides: &overrides})
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
}
```

`seedMemory` (in `update_memory_test.go`) needs `Global: true` added to the `SaveMemoryArgs` it constructs — see the implementation step.

- [ ] **Step 2: Run tests to verify they fail**

Run: `go build ./... && go test ./... && go test -tags integration ./internal/store/postgres/... -v`
Expected: many FAILs — `SaveMemoryArgs`/`UpdateMemoryArgs` have no `Global`/`Overrides` fields yet, and every existing `save_memory`-calling test that omits `Project` now hits the not-yet-implemented validation once it exists (chicken-and-egg with Step 3 — run this step's `go build` first to confirm the *new* tests fail to build for the expected reason, then proceed to Step 3, then return to Step 4 to see the *old* tests that need `Global: true` added actually fail correctly before you add it).

- [ ] **Step 3: Implement**

```go
// internal/store/postgres/notes.go — add Overrides to NewMemory
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
	Overrides      string
}
```

```go
// internal/store/postgres/notes.go — CreateMemory gains overrides resolution + write
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

	var overridesID any
	if m.Overrides != "" {
		id, err := idForSlug(ctx, tx, m.Overrides)
		if err != nil {
			return nil, err
		}
		overridesID = id
	}

	row := tx.QueryRow(ctx, `
		INSERT INTO memories (type, slug, description, body, project, agent, tags, embedding, embedding_model, overrides)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		RETURNING `+memoryColumns,
		m.Type, m.Slug, m.Description, m.Body, nullableString(m.Project), nullableString(m.Agent),
		tags, pgvector.NewVector(m.Embedding), m.EmbeddingModel, overridesID)

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
```

```go
// internal/store/postgres/notes.go — replace Task 3's applyOverridesPatch
// placeholder with the real implementation
func applyOverridesPatch(ctx context.Context, tx pgx.Tx, slug, overrides string) error {
	var overridesID any
	if overrides != "" {
		id, err := idForSlug(ctx, tx, overrides)
		if err != nil {
			return err
		}
		overridesID = id
	}
	_, err := tx.Exec(ctx, `UPDATE memories SET overrides = $1 WHERE slug = $2`, overridesID, slug)
	return err
}
```

(`UpdateMemory` itself is otherwise unchanged from Task 3 — it already calls `applyOverridesPatch` when `p.Overrides != nil`.)

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
	Project     string   `json:"project,omitempty" jsonschema:"project this memory relates to; required unless global is true"`
	Agent       string   `json:"agent,omitempty" jsonschema:"agent/client saving this, e.g. claude-code"`
	Tags        []string `json:"tags,omitempty" jsonschema:"arbitrary labels for filtering"`
	Global      bool     `json:"global,omitempty" jsonschema:"explicitly save without a project scope; required if project is omitted"`
	Overrides   string   `json:"overrides,omitempty" jsonschema:"slug of a global memory this project-scoped memory takes precedence over"`
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
	if a.Project == "" && !a.Global {
		return nil, SaveMemoryResult{}, fmt.Errorf("project is required unless global is explicitly set to true")
	}
	if err := t.validateOverrides(ctx, a.Project, a.Overrides); err != nil {
		return nil, SaveMemoryResult{}, err
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
		Overrides:      a.Overrides,
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
// internal/tools/update_memory.go — add Overrides to UpdateMemoryArgs and
// wire it into the patch, after the existing type/existing-fetch logic

type UpdateMemoryArgs struct {
	Slug              string    `json:"slug" jsonschema:"slug of the memory to update"`
	Type              *string   `json:"type,omitempty" jsonschema:"new memory type: user, feedback, project, or reference"`
	Description       *string   `json:"description,omitempty"`
	Body              *string   `json:"body,omitempty"`
	Project           *string   `json:"project,omitempty"`
	Agent             *string   `json:"agent,omitempty"`
	Tags              *[]string `json:"tags,omitempty"`
	Overrides         *string   `json:"overrides,omitempty" jsonschema:"slug of a global memory to override, or empty string to clear"`
	ClearSupersededBy bool      `json:"clear_superseded_by,omitempty"`
}
```

Add, right after the existing block that fetches `existing` and before `patch := postgres.MemoryPatch{...}` is returned/used — validate `Overrides` against the *resolved* project (the patch's new project if given, else the existing one):

```go
	if a.Overrides != nil && *a.Overrides != "" {
		project := existing.Project
		if a.Project != nil {
			project = *a.Project
		}
		if err := t.validateOverrides(ctx, project, *a.Overrides); err != nil {
			return nil, UpdateMemoryResult{}, err
		}
	}

	patch := postgres.MemoryPatch{
		Type:              a.Type,
		Description:       a.Description,
		Body:              a.Body,
		Project:           a.Project,
		Agent:             a.Agent,
		Tags:              a.Tags,
		Overrides:         a.Overrides,
		ClearSupersededBy: a.ClearSupersededBy,
	}
```

(This replaces the existing `patch := postgres.MemoryPatch{...}` literal — same fields as before, plus the two new ones.)

```go
// internal/tools/helpers.go — add validateOverrides
package tools

import (
	"context"
	"errors"
	"fmt"

	"github.com/lonhutt/recall/internal/models"
	"github.com/lonhutt/recall/internal/store/postgres"
)

// limitOrDefault resolves a caller-supplied result limit; absent (0) becomes
// def, and anything out of range gets pulled to the nearest bound.
func limitOrDefault(v, low, high, def int) int {
	switch {
	case v == 0:
		return def
	case v < low:
		return low
	case v > high:
		return high
	default:
		return v
	}
}

func memoryFilterFrom(typ, project, agent string, tags []string) postgres.MemoryFilter {
	f := postgres.MemoryFilter{Tags: tags}
	if typ != "" {
		f.Type = &typ
	}
	if project != "" {
		f.Project = &project
	}
	if agent != "" {
		f.Agent = &agent
	}
	return f
}

// validateOverrides checks that overrides (if set) is a legal
// project-overrides-global relationship: the declaring memory must be
// project-scoped, and the target must exist and be global.
func (t *Tools) validateOverrides(ctx context.Context, project, overrides string) error {
	if overrides == "" {
		return nil
	}
	if project == "" {
		return fmt.Errorf("overrides requires this memory to be project-scoped; global memories can't override anything")
	}
	target, err := t.Store.GetMemoryBySlug(ctx, overrides)
	if err != nil {
		if errors.Is(err, models.ErrNotFound) {
			return fmt.Errorf("overrides target %q not found", overrides)
		}
		return err
	}
	if target.Project != "" {
		return fmt.Errorf("overrides target %q is project-scoped (%q); overrides only applies to global memories", overrides, target.Project)
	}
	return nil
}
```

Finally, sweep every existing `SaveMemoryArgs{}` literal in `internal/tools/*_test.go` that doesn't already set `Project` and add `Global: true`: `save_memory_test.go` (`TestSaveMemorySucceeds`, `TestSaveMemoryChunksAnOversizedBody`, `TestSaveMemoryDuplicateSlugReturnsFriendlyError`, `TestSaveMemoryWrapsEmbeddingFailure`), `update_memory_test.go`'s `seedMemory` helper, `delete_memory_test.go`, `list_memories_test.go`, `e2e_test.go`, `backfill_test.go` doesn't call `SaveMemory` directly (it calls `BackfillMemoryEmbedding`, unaffected). Check each file for `SaveMemoryArgs{` and confirm it now compiles/passes with the new validation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `go build ./... && go vet ./... && go test ./...`
Expected: PASS across the whole non-integration suite.

Run: `go test -tags integration ./internal/store/postgres/... -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/store/postgres/notes.go internal/store/postgres/notes_test.go internal/tools/save_memory.go internal/tools/save_memory_test.go internal/tools/update_memory.go internal/tools/update_memory_test.go internal/tools/helpers.go internal/tools/delete_memory_test.go internal/tools/list_memories_test.go internal/tools/e2e_test.go
git commit -m "add overrides field and explicit project/global requirement to save_memory/update_memory"
```

---

## Task 5: Search ranking — supersession/override tiering + recency decay + age

**Files:**
- Modify: `internal/store/postgres/search.go`
- Modify: `internal/tools/search_memory.go`
- Modify: `internal/tools/dto.go`
- Modify: `internal/tools/helpers.go`
- Test: `internal/store/postgres/search_test.go`

**Interfaces:**
- Consumes: `models.Memory.SupersededBy`/`.Overrides`/`.Type`/`.UpdatedAt` (Tasks 2-4).
- Produces: `ScoredMemory.OverriddenBy string`; `humanAge(t time.Time) string` — used by `search_memory`'s DTO population.

> **Known defect (found 2026-10-31 in the hybrid-search review):** `TestSearchMemoriesDemotesOverriddenGlobalMemoryOnlyWithinTheOverridingProject` expects the global `global-style` back from a search scoped to `Project: "recall"`, but `buildFilter` emits `project = $n`, which excludes NULL-project rows. As written it fails with "expected both memories in scoped results". Decide before running this task whether project-scoped search should include globals (`(project = $n OR project IS NULL)`, in `buildFilter` or a search-only variant); every leg of hybrid search, keyword included, goes through `buildFilter`.

- [ ] **Step 1: Write the failing tests**

```go
// internal/store/postgres/search_test.go — append
func TestSearchMemoriesRanksSupersededMemoriesBelowCurrentOnes(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "old-answer", Description: "d", Body: "b", Embedding: vector(1, 0), EmbeddingModel: "embeddinggemma",
	}); err != nil {
		t.Fatalf("CreateMemory(old-answer): %v", err)
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "weak-match", Description: "d", Body: "b", Embedding: vector(0.1, 0.05), EmbeddingModel: "embeddinggemma",
	}); err != nil {
		t.Fatalf("CreateMemory(weak-match): %v", err)
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "new-answer", Description: "d", Body: "b", Embedding: vector(1, 0), EmbeddingModel: "embeddinggemma",
	}); err != nil {
		t.Fatalf("CreateMemory(new-answer): %v", err)
	}
	if _, err := store.SupersedeMemory(ctx, "old-answer", "new-answer"); err != nil {
		t.Fatalf("SupersedeMemory: %v", err)
	}

	model := "embeddinggemma"
	results, err := store.SearchMemories(ctx, vector(1, 0), "", postgres.MemoryFilter{EmbeddingModel: &model}, 10)
	if err != nil {
		t.Fatalf("SearchMemories: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3", len(results))
	}
	// old-answer scores identically to new-answer (same embedding) but must
	// rank below weak-match (a strictly worse cosine match) because it's
	// superseded — tier beats score.
	var oldIdx, weakIdx int = -1, -1
	for i, r := range results {
		switch r.Slug {
		case "old-answer":
			oldIdx = i
		case "weak-match":
			weakIdx = i
		}
	}
	if oldIdx < weakIdx {
		t.Errorf("old-answer (superseded) ranked above weak-match (not superseded) despite a lower-quality match; results = %+v", results)
	}
	for _, r := range results {
		if r.Slug == "old-answer" && r.SupersededBy != "new-answer" {
			t.Errorf("old-answer.SupersededBy = %q, want new-answer", r.SupersededBy)
		}
	}
}

func TestSearchMemoriesDemotesOverriddenGlobalMemoryOnlyWithinTheOverridingProject(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "global-style", Description: "d", Body: "b", Embedding: vector(1, 0), EmbeddingModel: "embeddinggemma",
	}); err != nil {
		t.Fatalf("CreateMemory(global-style): %v", err)
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "project", Slug: "project-style", Description: "d", Body: "b", Embedding: vector(0.9, 0.1),
		Project: "recall", Overrides: "global-style", EmbeddingModel: "embeddinggemma",
	}); err != nil {
		t.Fatalf("CreateMemory(project-style): %v", err)
	}

	model := "embeddinggemma"
	project := "recall"

	scoped, err := store.SearchMemories(ctx, vector(1, 0), "", postgres.MemoryFilter{EmbeddingModel: &model, Project: &project}, 10)
	if err != nil {
		t.Fatalf("SearchMemories (scoped): %v", err)
	}
	var globalIdx, projectIdx int = -1, -1
	for i, r := range scoped {
		switch r.Slug {
		case "global-style":
			globalIdx = i
		case "project-style":
			projectIdx = i
		}
		if r.Slug == "global-style" && r.OverriddenBy != "project-style" {
			t.Errorf("within recall's scope, global-style.OverriddenBy = %q, want project-style", r.OverriddenBy)
		}
	}
	if projectIdx == -1 || globalIdx == -1 {
		t.Fatalf("expected both memories in scoped results, got %+v", scoped)
	}
	if globalIdx < projectIdx {
		t.Errorf("global-style ranked above project-style despite being overridden in this project's scope; results = %+v", scoped)
	}

	unscoped, err := store.SearchMemories(ctx, vector(1, 0), "", postgres.MemoryFilter{EmbeddingModel: &model}, 10)
	if err != nil {
		t.Fatalf("SearchMemories (unscoped): %v", err)
	}
	for _, r := range unscoped {
		if r.Slug == "global-style" && r.OverriddenBy != "" {
			t.Errorf("outside recall's scope, global-style.OverriddenBy = %q, want empty (override doesn't apply here)", r.OverriddenBy)
		}
	}
}

func TestSearchMemoriesAppliesTypeSpecificRecencyDecay(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "user", Slug: "durable-pref", Description: "d", Body: "b", Embedding: vector(1, 0), EmbeddingModel: "embeddinggemma",
	}); err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	model := "embeddinggemma"
	results, err := store.SearchMemories(ctx, vector(1, 0), "", postgres.MemoryFilter{EmbeddingModel: &model}, 10)
	if err != nil {
		t.Fatalf("SearchMemories: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}
	// Similarity must stay the raw cosine (1.0 for an identical vector): decay
	// scales Score only.
	if results[0].Similarity < 0.999 {
		t.Errorf("Similarity = %v for an identical vector, want ~1.0 (decay must not touch it)", results[0].Similarity)
	}
}
```

A just-created memory can't show decay (the factor is 1 − ~1e-8), and the external test package can't backdate `updated_at`, so pin the half-lives with an internal-package unit test:

```go
// internal/store/postgres/decay_test.go — new, package postgres, no build tag
func TestDecayFactorHalvesAtEachTypesHalfLife(t *testing.T) {
	now := time.Now()
	cases := []struct {
		memType string
		age     time.Duration
	}{
		{"user", 720 * 24 * time.Hour},
		{"reference", 720 * 24 * time.Hour},
		{"project", 90 * 24 * time.Hour},
		{"feedback", 90 * 24 * time.Hour},
	}
	for _, c := range cases {
		if got := decayFactor(c.memType, now.Add(-c.age)); math.Abs(got-0.5) > 1e-3 {
			t.Errorf("decayFactor(%q, %v ago) = %v, want ~0.5", c.memType, c.age, got)
		}
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test -tags integration ./internal/store/postgres/... -run 'TestSearchMemoriesRanksSupersededMemoriesBelowCurrentOnes|TestSearchMemoriesDemotesOverriddenGlobalMemoryOnlyWithinTheOverridingProject|TestSearchMemoriesAppliesTypeSpecificRecencyDecay|TestDecayFactorHalvesAtEachTypesHalfLife' -v`
Expected: FAIL to build — `ScoredMemory.OverriddenBy` doesn't exist yet.

- [ ] **Step 3: Implement**

This is a delta on the hybrid `search.go` (Obsidian `recall/Plans/2026-10-31-hybrid-search-and-retrieval-eval.md`), not a replacement. Leave the two vector queries, the keyword leg and RRF fusion as they are; tiering and decay are a pass over the fused rows, after the row fetch and before the final sort and `limit` truncation, so they see the full `max(limit, 50)`-deep candidate set.

```go
// internal/store/postgres/search.go — additions
type ScoredMemory struct {
	models.Memory
	Score      float64 // unchanged doc comment
	Similarity float64 // unchanged doc comment
	// OverriddenBy is set only when this search was scoped to a project
	// (MemoryFilter.Project) and some memory in that project declares
	// overrides pointing at this one. Empty otherwise, even if this memory
	// is overridden somewhere else — the relationship is scope-conditional.
	OverriddenBy string
}

const (
	decayHalfLifeDurable  = 720 * 24 * time.Hour // user, reference
	decayHalfLifeVolatile = 90 * 24 * time.Hour  // project, feedback
)

// decayFactor is a [0,1] multiplier on a memory's fused score: 1.0 for a
// just-touched memory, halving every half-life's worth of age.
func decayFactor(memType string, updatedAt time.Time) float64

// overriddenWithinProject maps a memory's id to the slug of whichever memory
// in project declares overrides pointing at it.
func (s *Store) overriddenWithinProject(ctx context.Context, project string) (map[string]string, error)
```

`decayFactor` picks `decayHalfLifeDurable` for `user`/`reference` and `decayHalfLifeVolatile` otherwise, and returns `math.Exp(-math.Ln2 * age.Hours() / halfLife.Hours())` with `age = time.Since(updatedAt)`. `overriddenWithinProject` runs `SELECT overrides, slug FROM memories WHERE project = $1 AND overrides IS NOT NULL`.

Changes inside `SearchMemories`:

- In the row loop, `Score: fused[m.ID] * decayFactor(m.Type, m.UpdatedAt)`. Decay scales the fused rank value, never `Similarity`; `Similarity` stays the raw cosine so a caller can still tell a weak match from a stale one.
- After the loop, when `f.Project != nil`, fill `OverriddenBy` from `overriddenWithinProject`.
- The final sort becomes tier first (1 when `SupersededBy != ""` or `OverriddenBy != ""`, else 0, lower first), then the existing `Score` desc, `Similarity` desc, `ID` asc tie-breakers.
- Update the doc comment: ranking is tier, then decayed fused score.

Imports gain `"math"` and `"time"`.

```go
// internal/tools/helpers.go — add humanAge
import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/lonhutt/recall/internal/models"
	"github.com/lonhutt/recall/internal/store/postgres"
)

// humanAge renders how long ago t was, coarsely — search_memory results
// use this so an agent sees staleness at a glance instead of having to do
// timestamp arithmetic itself.
func humanAge(t time.Time) string {
	days := int(time.Since(t).Hours() / 24)
	switch {
	case days <= 0:
		return "today"
	case days == 1:
		return "1 day"
	default:
		return fmt.Sprintf("%d days", days)
	}
}
```

```go
// internal/tools/dto.go — add OverriddenBy and Age to ScoredMemoryDTO
type ScoredMemoryDTO struct {
	MemoryDTO
	Score        float64 `json:"score"`
	Similarity   float64 `json:"similarity"`
	OverriddenBy string  `json:"overridden_by,omitempty"`
	Age          string  `json:"age,omitempty"`
}
```

```go
// internal/tools/search_memory.go — update the result-building loop
	results := make([]ScoredMemoryDTO, len(scored))
	for i, s := range scored {
		results[i] = ScoredMemoryDTO{
			MemoryDTO:    newMemoryDTO(&s.Memory),
			Score:        s.Score,
			Similarity:   s.Similarity,
			OverriddenBy: s.OverriddenBy,
			Age:          humanAge(s.UpdatedAt),
		}
	}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test -tags integration ./internal/store/postgres/... -v`
Expected: PASS for all postgres integration tests, including the 3 new ones and every pre-existing search test (`TestSearchMemoriesOrdersByCosineSimilarity`, `TestSearchMemoriesFindsMatchViaBodyChunkNotJustDescription`, `TestSearchMemoriesDeduplicatesMultipleChunkHitsToOneRowPerMemory`, `TestSearchMemoriesToleratesAMemoryWithNoChunks`, `TestSearchMemoriesHybridFindsExactTermTheVectorRanksLow`, `TestSearchMemoriesReportsSimilarity`, and the rest) — these all create memories moments before searching, so decay is negligible and shouldn't shift any of their existing ordering assertions.

Run: `go build ./... && go vet ./... && go test ./...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/store/postgres/search.go internal/store/postgres/search_test.go internal/tools/search_memory.go internal/tools/dto.go internal/tools/helpers.go
git commit -m "rank search results by supersession/override tier and recency decay"
```

---

## Task 6: Write-time duplicate detection

> **Note (2026-10-31):** hybrid search doesn't change this task. `FindSimilarMemory` runs its own single cosine query against `memories.embedding` and never goes through `SearchMemories` or RRF, so its `Score` stays a raw cosine compared against the 0.92 threshold.

**Files:**
- Modify: `internal/store/postgres/search.go`
- Modify: `internal/tools/deps.go`
- Modify: `internal/tools/save_memory.go`
- Modify: `internal/tools/fakes_test.go`
- Test: `internal/store/postgres/search_test.go`
- Test: `internal/tools/save_memory_test.go`

**Interfaces:**
- Consumes: `memoryColumns`/`scanMemory` (Task 2), `descVec` already computed in `save_memory.go` (existing).
- Produces: `Store.FindSimilarMemory(ctx, embedding []float32, memType, project string, threshold float64) (*ScoredMemory, error)`.

- [ ] **Step 1: Write the failing tests**

```go
// internal/store/postgres/search_test.go — append
func TestFindSimilarMemoryPrefersProjectScopeThenFallsBackToGlobal(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "user", Slug: "global-similar", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(global-similar): %v", err)
	}

	// No project-scoped user-type memory exists yet — falls back to global.
	hit, err := store.FindSimilarMemory(ctx, vector(1, 0), "user", "recall", 0.9)
	if err != nil {
		t.Fatalf("FindSimilarMemory: %v", err)
	}
	if hit == nil || hit.Slug != "global-similar" {
		t.Fatalf("hit = %+v, want the global memory found via fallback", hit)
	}

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "user", Slug: "project-similar", Description: "d", Body: "b", Embedding: vector(1, 0), Project: "recall",
	}); err != nil {
		t.Fatalf("CreateMemory(project-similar): %v", err)
	}

	hit, err = store.FindSimilarMemory(ctx, vector(1, 0), "user", "recall", 0.9)
	if err != nil {
		t.Fatalf("FindSimilarMemory: %v", err)
	}
	if hit == nil || hit.Slug != "project-similar" {
		t.Fatalf("hit = %+v, want the project-scoped memory to win once one exists", hit)
	}
}

func TestFindSimilarMemoryIgnoresOtherTypesAndSupersededMemories(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "feedback", Slug: "wrong-type", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(wrong-type): %v", err)
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "user", Slug: "will-be-superseded", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(will-be-superseded): %v", err)
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "user", Slug: "supersedes-it", Description: "d", Body: "b", Embedding: vector(0, 1),
	}); err != nil {
		t.Fatalf("CreateMemory(supersedes-it): %v", err)
	}
	if _, err := store.SupersedeMemory(ctx, "will-be-superseded", "supersedes-it"); err != nil {
		t.Fatalf("SupersedeMemory: %v", err)
	}

	hit, err := store.FindSimilarMemory(ctx, vector(1, 0), "user", "", 0.9)
	if err != nil {
		t.Fatalf("FindSimilarMemory: %v", err)
	}
	if hit != nil {
		t.Errorf("hit = %+v, want nil — the only user-type candidate is superseded, and the feedback-type one doesn't count", hit)
	}
}

func TestFindSimilarMemoryReturnsNilBelowThreshold(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "user", Slug: "unrelated", Description: "d", Body: "b", Embedding: vector(0, 1),
	}); err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	hit, err := store.FindSimilarMemory(ctx, vector(1, 0), "user", "", 0.9)
	if err != nil {
		t.Fatalf("FindSimilarMemory: %v", err)
	}
	if hit != nil {
		t.Errorf("hit = %+v, want nil — orthogonal vectors score ~0, well under threshold", hit)
	}
}
```

```go
// internal/tools/save_memory_test.go — append
func TestSaveMemoryBlocksOnALikelyDuplicate(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	if _, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "user", Slug: "first", Description: "likes tabs", Body: "b", Global: true,
	}); err != nil {
		t.Fatalf("seeding: %v", err)
	}
	store.similarResp = &postgres.ScoredMemory{Memory: models.Memory{Slug: "first"}, Score: 0.95}

	_, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "user", Slug: "second", Description: "prefers tabs", Body: "b", Global: true,
	})
	if err == nil {
		t.Fatal("expected an error for a likely duplicate, got nil")
	}
	if store.bySlug["second"].Slug != "" {
		t.Error("SaveMemory must not have created the memory when blocked")
	}
}

func TestSaveMemoryForceBypassesTheDuplicateCheck(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	store.similarResp = &postgres.ScoredMemory{Memory: models.Memory{Slug: "first"}, Score: 0.95}

	_, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "user", Slug: "second", Description: "prefers tabs", Body: "b", Global: true, Force: true,
	})
	if err != nil {
		t.Fatalf("SaveMemory with Force: true: %v", err)
	}
}

func TestSaveMemoryProceedsWhenNoSimilarMemoryFound(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}

	_, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "user", Slug: "s", Description: "d", Body: "b", Global: true,
	})
	if err != nil {
		t.Fatalf("SaveMemory: %v", err)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test -tags integration ./internal/store/postgres/... -run TestFindSimilarMemory -v` and `go test ./internal/tools/... -run 'TestSaveMemoryBlocksOnALikelyDuplicate|TestSaveMemoryForceBypassesTheDuplicateCheck|TestSaveMemoryProceedsWhenNoSimilarMemoryFound' -v`
Expected: FAIL to build — `Store.FindSimilarMemory`, `SaveMemoryArgs.Force`, `fakeStore.similarResp` don't exist yet.

- [ ] **Step 3: Implement**

```go
// internal/store/postgres/search.go — append
// FindSimilarMemory returns the closest live (non-superseded) memory of
// memType at or above threshold — checking project first if project is
// non-empty, falling back to global memories of the same type if nothing
// scores high enough there — or nil if nothing qualifies anywhere. This is
// the write-time duplicate check (save_memory): single-phase,
// description-embedding only, one candidate, not the full two-phase search.
func (s *Store) FindSimilarMemory(ctx context.Context, embedding []float32, memType, project string, threshold float64) (*ScoredMemory, error) {
	vec := pgvector.NewVector(embedding)

	tryScope := func(extraWhere string, args []any) (*ScoredMemory, error) {
		query := fmt.Sprintf(`
			SELECT %s, 1 - (embedding <=> $1) AS score
			FROM memories
			WHERE type = $2 AND superseded_by IS NULL %s
			ORDER BY embedding <=> $1
			LIMIT 1
		`, memoryColumns, extraWhere)
		var score float64
		m, err := scanMemory(s.pool.QueryRow(ctx, query, args...), &score)
		if errors.Is(err, models.ErrNotFound) {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}
		if score < threshold {
			return nil, nil
		}
		return &ScoredMemory{Memory: *m, Score: score}, nil
	}

	if project != "" {
		hit, err := tryScope("AND project = $3", []any{vec, memType, project})
		if err != nil || hit != nil {
			return hit, err
		}
	}
	return tryScope("AND project IS NULL", []any{vec, memType})
}
```

Add `"errors"` back to `search.go`'s import block (needed here for `errors.Is(err, models.ErrNotFound)`).

```go
// internal/tools/deps.go — add FindSimilarMemory to the Store interface
	FindSimilarMemory(ctx context.Context, embedding []float32, memType, project string, threshold float64) (*postgres.ScoredMemory, error)
```

```go
// internal/tools/save_memory.go — add near the top of the file, after imports
// duplicateSimilarityThreshold is a deliberately hardcoded corner cut — a
// tune-once-you-have-real-usage-data value, not exposed as config yet.
const duplicateSimilarityThreshold = 0.92
```

Insert the duplicate check in `SaveMemory`, right after `descVec, err := t.embedDescription(ctx, a.Description)` succeeds and before `chunkAndEmbedBody` is called (no reason to spend an embed call chunking a body that might get blocked anyway):

```go
	if !a.Force {
		similar, err := t.Store.FindSimilarMemory(ctx, descVec, a.Type, a.Project, duplicateSimilarityThreshold)
		if err != nil {
			return nil, SaveMemoryResult{}, fmt.Errorf("checking for similar memories: %w", err)
		}
		if similar != nil {
			scope := "global"
			if similar.Project != "" {
				scope = fmt.Sprintf("project %q", similar.Project)
			}
			return nil, SaveMemoryResult{}, fmt.Errorf(
				"a similar memory already exists: %q (%s, score %.2f); use update_memory to modify it, or pass force: true to save anyway",
				similar.Slug, scope, similar.Score,
			)
		}
	}
```

And add `Force bool` to `SaveMemoryArgs`:

```go
	Force bool `json:"force,omitempty" jsonschema:"bypass the near-duplicate check and save anyway"`
```

```go
// internal/tools/fakes_test.go — add to fakeStore
	similarResp *postgres.ScoredMemory

func (f *fakeStore) FindSimilarMemory(_ context.Context, _ []float32, _, _ string, _ float64) (*postgres.ScoredMemory, error) {
	return f.similarResp, nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test -tags integration ./internal/store/postgres/... -v`
Expected: PASS.

Run: `go build ./... && go vet ./... && go test ./...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/store/postgres/search.go internal/store/postgres/search_test.go internal/tools/deps.go internal/tools/save_memory.go internal/tools/save_memory_test.go internal/tools/fakes_test.go
git commit -m "block save_memory on a likely duplicate, bypassable with force"
```

---

## Task 7: Recall skill instructions — scope derivation + duplicate-collision resolution

**Files:**
- Modify: `.claude/skills/recall/SKILL.md`

**Interfaces:**
- Consumes: the `global`/`overrides`/`force` args (Tasks 4, 6) and the error messages `save_memory` now returns.
- Produces: no code — this is the calling-agent-side instructions those server-side changes depend on to be used correctly (per the spec's explicit skill-decides/server-enforces split).

This task has no automated test — it's prose, reviewed for accuracy against the spec and the actual tool behavior from Tasks 1-6, not `go test`.

- [ ] **Step 1: Update the "When MCP tools are already connected" section**

Replace the existing bullet list (the one starting "`project`: the absolute path...") with:

```markdown
- `project`: the absolute path of the project the memory relates to (e.g.
  `/home/lhutt/Projects/dcx`) — **required on every `save_memory` call
  unless the memory is genuinely global**, in which case pass `global: true`
  instead. Derive it before calling `save_memory`, don't ask the caller to
  supply it separately: walk up from the current working directory looking
  for `CLAUDE.md`, `AGENTS.md`, or a build file (`go.mod`, `package.json`,
  `pom.xml`, `Cargo.toml`, `Makefile`, etc.) — the directory containing the
  first one found is the project root, use its absolute path. If the
  working directory is `/`, directly under `/home/*` with no such markers,
  or otherwise clearly not inside a project, pass `global: true` instead.
  `save_memory` rejects a call that supplies neither — it never silently
  guesses.
- `agent`: `"claude-code"`.
- `type`: one of `user` / `feedback` / `project` / `reference` — same
  taxonomy the file-based memory system used, so the same judgment about
  what's worth saving applies (surprising, non-obvious, not derivable from
  code or git history — not routine facts).
- Before saving something that might already be known, `search_memory` or
  `list_memories` first — cheaper than a duplicate, and `update_memory` is
  the right tool if it turns out to already exist (patches only given
  fields, re-embeds only if description/body changed).
- If a project-scoped memory is meant to take precedence over an existing
  global one on the same topic (not replace it — both stay true, just in
  different scopes), pass `overrides: <the global memory's slug>` on
  `save_memory`/`update_memory` rather than just letting them coexist
  unrelated. `search_memory` uses this to demote the global one within that
  project's results instead of returning both as if they were equally
  current.
- If a memory is flatly wrong now — not "true in a narrower scope," just
  outdated — use `supersede_memory(old_slug, new_slug)` after saving the
  replacement, not `update_memory` on the old one (that would erase the
  historical record; supersession keeps it, just demoted and marked).
```

- [ ] **Step 2: Add a new section on the duplicate-block collision flow**

Insert after the section from Step 1, before "## When they're not connected":

```markdown
## When `save_memory` blocks on a likely duplicate

`save_memory` checks for an existing memory that looks like a near-duplicate
before creating a new one, and blocks with an error naming the existing
slug, its scope (project or global), and a similarity score if it finds
one. **Don't silently retry with `force: true` and don't silently give up —
stop and ask the person you're working with**, offering exactly these three
options:

1. **Save as project-specific** — re-save scoped to the current project;
   if the collision was against a *global* memory, also pass
   `overrides: <the existing slug>` (this is the override relationship from
   the section above — the new content is true for this project specifically,
   the old global fact isn't necessarily wrong elsewhere).
2. **Save as new global** — re-save with `force: true`, then call
   `supersede_memory(existing_slug, new_slug)` — the new content replaces
   the old fact rather than coexisting with it.
3. **Drop it** — make no further calls; the existing memory already covers
   it.

This exists because a near-duplicate save is exactly how a memory vault
quietly fills up with redundant or contradicting entries — the block is
deliberate friction, not a bug to route around.
```

- [ ] **Step 3: Self-review the skill changes**

Re-read the full `SKILL.md` after both edits. Confirm: no contradiction with the "Operating the stack"/"Schema migrations" sections below (they're unaffected — this task only touches the "already connected" tool-usage guidance), the new project-derivation instructions match exactly what Task 4's `save_memory` validation actually enforces (test this against Task 4's `TestSaveMemoryRequiresProjectOrExplicitGlobal` if there's any doubt about the exact error condition), and the three collision-resolution options match Task 6's actual error message shape (`"a similar memory already exists: %q (%s, score %.2f); use update_memory to modify it, or pass force: true to save anyway"`).

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/recall/SKILL.md
git commit -m "document scope derivation and duplicate-collision resolution in the recall skill"
```

---

## Final verification (after all 7 tasks)

Run, in order:

1. `gofmt -l .` — expect no output.
2. `go vet ./...` — expect no output.
3. `go build ./...` — expect success.
4. `go test ./...` — expect all packages `ok`.
5. `docker ps` to confirm Docker is reachable, then `go test -tags integration ./...` — expect all packages `ok`.
6. Manually re-read `.claude/skills/recall/SKILL.md` end to end once more, since Task 7 has no automated check of its own — confirm it reads coherently as a whole document, not just as two isolated edits.

Do not rebuild/redeploy the live `recall` container or run any live-database operation as part of this verification — that's a deliberate, separate step for the person running this plan to take (and confirm with the user first), same as the chunked-embeddings plan before it.
