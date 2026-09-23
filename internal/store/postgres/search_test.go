//go:build integration

package postgres_test

import (
	"context"
	"testing"

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

	results, err := store.SearchMemories(ctx, vector(0.9, 0.1), postgres.MemoryFilter{}, 10)
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

	alpha := "alpha"
	results, err := store.SearchMemories(ctx, vector(1), postgres.MemoryFilter{Project: &alpha}, 10)
	if err != nil {
		t.Fatalf("SearchMemories: %v", err)
	}
	for _, r := range results {
		if r.Project != "alpha" {
			t.Errorf("SearchMemories with project filter returned %q", r.Project)
		}
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
	results, err := store.SearchMemories(ctx, vector(1), postgres.MemoryFilter{EmbeddingModel: &model}, 10)
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
