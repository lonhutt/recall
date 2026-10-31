//go:build integration

package postgres_test

import (
	"context"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/lonhutt/recall/internal/store/postgres"
)

func TestSearchMemoriesOrdersByCosineSimilarity(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	create := func(slug string, embedding []float32) {
		_, err := store.CreateMemory(ctx, postgres.NewMemory{
			Type: "reference", Slug: slug, Description: "d", Body: "b", Embedding: embedding,
		})
		if err != nil {
			t.Fatalf("CreateMemory(%s): %v", slug, err)
		}
	}
	create("close-to-query", vector(1, 0))
	create("far-from-query", vector(0, 1))

	results, err := store.SearchMemories(ctx, vector(0.9, 0.1), "", postgres.MemoryFilter{}, 10)
	if err != nil {
		t.Fatalf("SearchMemories: %v", err)
	}
	if len(results) < 2 {
		t.Fatalf("got %d results, want at least 2", len(results))
	}
	if results[0].Slug != "close-to-query" {
		t.Errorf("top result = %q, want close-to-query", results[0].Slug)
	}
	if results[0].Score <= results[1].Score {
		t.Errorf("top score %v not greater than next score %v", results[0].Score, results[1].Score)
	}
}

func TestSearchMemoriesFiltersByProject(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	_, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "project-a-note", Description: "d", Body: "b",
		Project: "alpha", Embedding: vector(1),
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}
	_, err = store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "project-b-note", Description: "d", Body: "b",
		Project: "beta", Embedding: vector(1),
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	_, err = store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "global-note", Description: "d", Body: "b", Embedding: vector(1),
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	alpha := "alpha"
	results, err := store.SearchMemories(ctx, vector(1), "", postgres.MemoryFilter{Project: &alpha}, 10)
	if err != nil {
		t.Fatalf("SearchMemories: %v", err)
	}
	var slugs []string
	for _, r := range results {
		slugs = append(slugs, r.Slug)
	}
	slices.Sort(slugs)
	if !slices.Equal(slugs, []string{"global-note", "project-a-note"}) {
		t.Errorf("results = %v, want alpha's memory and the global one, not beta's", slugs)
	}
}

func TestSearchMemoriesExcludesOtherEmbeddingModels(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	for _, row := range []struct{ slug, model string }{
		{"current-model-note", "embeddinggemma"},
		{"stale-model-note", "voyage-3-large"},
	} {
		_, err := store.CreateMemory(ctx, postgres.NewMemory{
			Type: "reference", Slug: row.slug, Description: "d", Body: "b",
			Embedding: vector(1), EmbeddingModel: row.model,
		})
		if err != nil {
			t.Fatalf("CreateMemory(%s): %v", row.slug, err)
		}
	}

	model := "embeddinggemma"
	results, err := store.SearchMemories(ctx, vector(1), "", postgres.MemoryFilter{EmbeddingModel: &model}, 10)
	if err != nil {
		t.Fatalf("SearchMemories: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d results, want only the row embedded by the current model", len(results))
	}
	if results[0].Slug != "current-model-note" {
		t.Errorf("result = %q, want current-model-note", results[0].Slug)
	}
}

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

	results, err := store.SearchMemories(ctx, vector(1, 0), "", postgres.MemoryFilter{}, 10)
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

	results, err := store.SearchMemories(ctx, vector(1, 0), "", postgres.MemoryFilter{}, 10)
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

	results, err := store.SearchMemories(ctx, vector(1, 0), "", postgres.MemoryFilter{}, 10)
	if err != nil {
		t.Fatalf("SearchMemories: %v", err)
	}
	if len(results) != 1 || results[0].Slug != "no-chunks" {
		t.Errorf("results = %+v, want the chunkless memory found via its description score", results)
	}
}

// createSearchable is the shared setup for the hybrid tests: one memory
// with a description embedding and body, under the current model.
func createSearchable(t *testing.T, store *postgres.Store, slug, project, body string, embedding []float32) {
	t.Helper()
	_, err := store.CreateMemory(context.Background(), postgres.NewMemory{
		Type: "reference", Slug: slug, Description: "d", Body: body, Project: project,
		Embedding: embedding, EmbeddingModel: "embeddinggemma",
	})
	if err != nil {
		t.Fatalf("CreateMemory(%s): %v", slug, err)
	}
}

func searchSlugs(t *testing.T, store *postgres.Store, embedding []float32, query string, f postgres.MemoryFilter) []string {
	t.Helper()
	model := "embeddinggemma"
	f.EmbeddingModel = &model
	results, err := store.SearchMemories(context.Background(), embedding, query, f, 10)
	if err != nil {
		t.Fatalf("SearchMemories(%q): %v", query, err)
	}
	slugs := make([]string, len(results))
	for i, r := range results {
		slugs[i] = r.Slug
	}
	return slugs
}

func TestSearchMemoriesHybridFindsExactTermTheVectorRanksLow(t *testing.T) {
	store := newTestStore(t)
	createSearchable(t, store, "env-note", "", "set RECALL_HTTP_TOKEN in .env", vector(0, 1))
	createSearchable(t, store, "distractor", "", "unrelated", vector(1, 0))

	if got := searchSlugs(t, store, vector(1, 0), "RECALL_HTTP_TOKEN", postgres.MemoryFilter{}); len(got) == 0 || got[0] != "env-note" {
		t.Errorf("hybrid results = %v, want env-note first", got)
	}
	if got := searchSlugs(t, store, vector(1, 0), "", postgres.MemoryFilter{}); len(got) == 0 || got[0] != "distractor" {
		t.Errorf("vector-only results = %v, want distractor first", got)
	}
}

func TestSearchMemoriesHybridUsesOrSemanticsForNaturalLanguage(t *testing.T) {
	store := newTestStore(t)
	createSearchable(t, store, "a-other", "", "tabs versus spaces", vector(1, 0))
	createSearchable(t, store, "z-commit-style", "", "keep commits to one subject line", vector(0.95, 0.05))

	got := searchSlugs(t, store, vector(1, 0), "how should git commits be structured", postgres.MemoryFilter{})
	if len(got) == 0 || got[0] != "z-commit-style" {
		t.Errorf("results = %v, want z-commit-style first: a query sharing only some words must still match", got)
	}
}

func TestSearchMemoriesToleratesQueriesWithNoLexemesOrOperatorCharacters(t *testing.T) {
	store := newTestStore(t)
	createSearchable(t, store, "first", "", "alpha", vector(1, 0))
	createSearchable(t, store, "second", "", "beta", vector(0, 1))

	want := searchSlugs(t, store, vector(1, 0), "", postgres.MemoryFilter{})
	for _, q := range []string{"the of and", "&|!( :*", "   "} {
		if got := searchSlugs(t, store, vector(1, 0), q, postgres.MemoryFilter{}); !slices.Equal(got, want) {
			t.Errorf("query %q: results = %v, want vector order %v", q, got, want)
		}
	}
}

func TestSearchMemoriesKeywordLegHonorsProjectFilter(t *testing.T) {
	store := newTestStore(t)
	createSearchable(t, store, "alpha-note", "alpha", "generic", vector(1))
	createSearchable(t, store, "beta-note", "beta", "zanzibar", vector(1))

	alpha := "alpha"
	got := searchSlugs(t, store, vector(1), "zanzibar", postgres.MemoryFilter{Project: &alpha})
	if !slices.Equal(got, []string{"alpha-note"}) {
		t.Errorf("results = %v, want only alpha-note", got)
	}
}

func TestSearchMemoriesKeywordLegExcludesOtherEmbeddingModels(t *testing.T) {
	store := newTestStore(t)
	_, err := store.CreateMemory(context.Background(), postgres.NewMemory{
		Type: "reference", Slug: "stale-note", Description: "d", Body: "zanzibar",
		Embedding: vector(1), EmbeddingModel: "voyage-3-large",
	})
	if err != nil {
		t.Fatalf("CreateMemory(stale-note): %v", err)
	}
	createSearchable(t, store, "current-note", "", "generic", vector(1))

	got := searchSlugs(t, store, vector(1), "zanzibar", postgres.MemoryFilter{})
	if !slices.Equal(got, []string{"current-note"}) {
		t.Errorf("results = %v, want only current-note", got)
	}
}

func TestSearchMemoriesReportsSimilarity(t *testing.T) {
	store := newTestStore(t)
	createSearchable(t, store, "only", "", "b", vector(1, 0))

	model := "embeddinggemma"
	results, err := store.SearchMemories(context.Background(), vector(1, 0), "anything", postgres.MemoryFilter{EmbeddingModel: &model}, 10)
	if err != nil {
		t.Fatalf("SearchMemories: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}
	if math.Abs(results[0].Similarity-1.0) > 1e-6 {
		t.Errorf("Similarity = %v, want ~1.0", results[0].Similarity)
	}
	if results[0].Score >= 0.05 {
		t.Errorf("Score = %v, want a fused rank value well below 0.05", results[0].Score)
	}
}

func TestSearchMemoriesReportsChunkSimilarityForKeywordOnlyMatch(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	// 50 fillers fill both vector legs' depth, so the target comes back
	// through the keyword leg alone.
	for i := range 50 {
		if _, err := store.CreateMemory(ctx, postgres.NewMemory{
			Type: "reference", Slug: fmt.Sprintf("filler-%d", i), Description: "d", Body: "filler",
			Embedding: vector(1, 0), EmbeddingModel: "embeddinggemma",
			Chunks: []postgres.MemoryChunk{{ChunkIndex: 0, Text: "filler", Embedding: vector(1, 0)}},
		}); err != nil {
			t.Fatalf("CreateMemory(filler-%d): %v", i, err)
		}
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "target", Description: "d", Body: "zanzibar",
		Embedding: vector(0, 1), EmbeddingModel: "embeddinggemma",
		Chunks: []postgres.MemoryChunk{{ChunkIndex: 0, Text: "zanzibar", Embedding: vector(0.6, 0.8)}},
	}); err != nil {
		t.Fatalf("CreateMemory(target): %v", err)
	}

	model := "embeddinggemma"
	results, err := store.SearchMemories(ctx, vector(1, 0), "zanzibar", postgres.MemoryFilter{EmbeddingModel: &model}, 10)
	if err != nil {
		t.Fatalf("SearchMemories: %v", err)
	}
	for _, r := range results {
		if r.Slug == "target" {
			if math.Abs(r.Similarity-0.6) > 1e-6 {
				t.Errorf("target Similarity = %v, want ~0.6 (its best chunk)", r.Similarity)
			}
			return
		}
	}
	t.Fatalf("target missing from results")
}

func TestCreateMemoryAcceptsBodyTooLargeForAFullTsvector(t *testing.T) {
	store := newTestStore(t)
	var body strings.Builder
	for i := 0; body.Len() < 1_500_000; i++ {
		fmt.Fprintf(&body, "w%d ", i)
	}
	createSearchable(t, store, "huge", "", body.String(), vector(1, 0))

	got := searchSlugs(t, store, vector(0, 1), "w5", postgres.MemoryFilter{})
	if !slices.Contains(got, "huge") {
		t.Errorf("results = %v, want huge found by a term near the start of its body", got)
	}
}

func TestSearchMemoriesMatchesExactTermsContainingAmpersands(t *testing.T) {
	store := newTestStore(t)
	createSearchable(t, store, "filler", "", "nothing", vector(1, 0))
	createSearchable(t, store, "decoy", "", "http://x.com/zzz x.com", vector(0.8, 0.6))
	createSearchable(t, store, "target", "", "http://x.com/a&b?c=1", vector(0, 1))

	got := searchSlugs(t, store, vector(1, 0), "http://x.com/a&b?c=1", postgres.MemoryFilter{})
	if len(got) == 0 || got[0] != "target" {
		t.Errorf("results = %v, want target first: only it contains the full URL", got)
	}
}

func TestSearchMemoriesFindsFilteredMatchesBeyondTheVectorIndexsCandidateList(t *testing.T) {
	// Seq scans off forces the HNSW index (embedding_model has no index of its
	// own), whose default scan stops at 40 candidates before the filter runs.
	store := connectTestStore(t, newTestDB(t)+"&options=-c%20enable_seqscan%3Doff")
	for i := range 50 {
		if _, err := store.CreateMemory(context.Background(), postgres.NewMemory{
			Type: "reference", Slug: fmt.Sprintf("old-model-%d", i), Description: "d", Body: "b",
			// distinct vectors: HNSW folds identical ones into a single element
			Embedding: vector(1, float32(i)*0.01), EmbeddingModel: "old-model",
		}); err != nil {
			t.Fatalf("CreateMemory(old-model-%d): %v", i, err)
		}
	}
	for i := range 3 {
		createSearchable(t, store, fmt.Sprintf("current-%d", i), "", "b", vector(0, 1))
	}

	got := searchSlugs(t, store, vector(1, 0), "", postgres.MemoryFilter{})
	if len(got) != 3 {
		t.Errorf("results = %v, want all 3 memories embedded by the current model", got)
	}
}

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

func TestFindSimilarMemoryPrefersProjectScopeThenFallsBackToGlobal(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "user", Slug: "global-similar", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(global-similar): %v", err)
	}

	// No project-scoped user-type memory exists yet — falls back to global.
	hit, err := store.FindSimilarMemory(ctx, vector(1, 0), "", "user", "recall", 0.9)
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

	hit, err = store.FindSimilarMemory(ctx, vector(1, 0), "", "user", "recall", 0.9)
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

	hit, err := store.FindSimilarMemory(ctx, vector(1, 0), "", "user", "", 0.9)
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

	hit, err := store.FindSimilarMemory(ctx, vector(1, 0), "", "user", "", 0.9)
	if err != nil {
		t.Fatalf("FindSimilarMemory: %v", err)
	}
	if hit != nil {
		t.Errorf("hit = %+v, want nil — orthogonal vectors score ~0, well under threshold", hit)
	}
}

func TestFindSimilarMemoryIgnoresOtherEmbeddingModels(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "user", Slug: "old-model", Description: "d", Body: "b", Embedding: vector(1, 0), EmbeddingModel: "old",
	}); err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	hit, err := store.FindSimilarMemory(ctx, vector(1, 0), "new", "user", "", 0.9)
	if err != nil {
		t.Fatalf("FindSimilarMemory: %v", err)
	}
	if hit != nil {
		t.Errorf("hit = %+v, want nil — a vector from another model isn't comparable", hit)
	}
}

// backdate sets slug's updated_at age into the past, which nothing in the
// store API can do.
func backdate(t *testing.T, connString, slug string, age time.Duration) {
	t.Helper()
	pool, err := pgxpool.New(context.Background(), connString)
	if err != nil {
		t.Fatalf("connecting: %v", err)
	}
	defer pool.Close()
	if _, err := pool.Exec(context.Background(), `UPDATE memories SET updated_at = now() - $1::interval WHERE slug = $2`,
		fmt.Sprintf("%d seconds", int(age.Seconds())), slug); err != nil {
		t.Fatalf("backdating %s: %v", slug, err)
	}
}

func TestSearchMemoriesStrongMatchSurvivesModestAge(t *testing.T) {
	connString := newTestDB(t)
	store := connectTestStore(t, connString)
	createSearchable(t, store, "strong-old", "", "b", vector(1, 0))
	createSearchable(t, store, "weak-fresh", "", "b", vector(0.5, 0.866))
	// Ten days barely dents a 90-day half-life; the far better match must win.
	backdate(t, connString, "strong-old", 10*24*time.Hour)
	if err := setType(connString, "strong-old", "project"); err != nil {
		t.Fatal(err)
	}

	got := searchSlugs(t, store, vector(1, 0), "", postgres.MemoryFilter{})
	if len(got) == 0 || got[0] != "strong-old" {
		t.Errorf("results = %v, want strong-old first", got)
	}
}

func TestSearchMemoriesDecayDemotesAMuchOlderSlightlyBetterMatch(t *testing.T) {
	connString := newTestDB(t)
	store := connectTestStore(t, connString)
	createSearchable(t, store, "better-ancient", "", "b", vector(0.9, 0.436))
	createSearchable(t, store, "good-fresh", "", "b", vector(0.8, 0.6))
	backdate(t, connString, "better-ancient", 180*24*time.Hour)
	if err := setType(connString, "better-ancient", "project"); err != nil {
		t.Fatal(err)
	}

	got := searchSlugs(t, store, vector(1, 0), "", postgres.MemoryFilter{})
	if len(got) == 0 || got[0] != "good-fresh" {
		t.Errorf("results = %v, want good-fresh first: two half-lives of age outweigh a 0.1 similarity edge", got)
	}
}

func setType(connString, slug, memType string) error {
	pool, err := pgxpool.New(context.Background(), connString)
	if err != nil {
		return err
	}
	defer pool.Close()
	_, err = pool.Exec(context.Background(), `UPDATE memories SET type = $1 WHERE slug = $2`, memType, slug)
	return err
}

func TestSearchMemoriesIgnoresOverridesFromSupersededMemories(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	createSearchable(t, store, "use-spaces", "", "b", vector(1, 0))
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "project", Slug: "use-tabs-here", Description: "d", Body: "b", Embedding: vector(1, 0),
		EmbeddingModel: "embeddinggemma", Project: "recall", Overrides: "use-spaces",
	}); err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}
	if _, err := store.SupersedeMemory(ctx, "use-tabs-here", "use-spaces"); err != nil {
		t.Fatalf("SupersedeMemory: %v", err)
	}

	model, project := "embeddinggemma", "recall"
	results, err := store.SearchMemories(ctx, vector(1, 0), "", postgres.MemoryFilter{EmbeddingModel: &model, Project: &project}, 10)
	if err != nil {
		t.Fatalf("SearchMemories: %v", err)
	}
	if len(results) == 0 || results[0].Slug != "use-spaces" || results[0].OverriddenBy != "" {
		t.Errorf("results = %+v, want use-spaces first and not overridden: its overrider is superseded", results)
	}
}
