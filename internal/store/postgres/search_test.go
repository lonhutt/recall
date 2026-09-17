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
