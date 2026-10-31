package tools

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/lonhutt/recall/internal/embeddings"
	"github.com/lonhutt/recall/internal/models"
	"github.com/lonhutt/recall/internal/store/postgres"
)

func TestSaveMemorySucceeds(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1, 2, 3}}
	tools := &Tools{Store: store, Embedder: embedder, EmbeddingModel: "embeddinggemma", MaxTokensPerChunk: 2048}

	_, result, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "user", Slug: "likes-go", Description: "likes Go", Body: "prefers Go for backend work", Global: true,
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
	if got := store.bySlug["likes-go"].EmbeddingModel; got != "embeddinggemma" {
		t.Errorf("stored EmbeddingModel = %q, want %q — search_memory filters on it", got, "embeddinggemma")
	}
}

func TestSaveMemoryChunksAnOversizedBody(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1, 2, 3}}
	tools := &Tools{Store: store, Embedder: embedder, MaxTokensPerChunk: 5}

	longBody := "one two three four five six seven eight nine ten eleven twelve"
	_, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "user", Slug: "long-memory", Description: "a long one", Body: longBody, Global: true,
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

func TestSaveMemoryRejectsUnknownType(t *testing.T) {
	tools := &Tools{Store: newFakeStore(), Embedder: &fakeEmbedder{vector: []float32{1}}}

	_, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "bogus", Slug: "s", Description: "d", Body: "b", Global: true,
	})
	if err == nil {
		t.Fatal("expected an error for unknown type, got nil")
	}
	if !strings.Contains(err.Error(), "bogus") {
		t.Errorf("error %q does not mention the offending type", err.Error())
	}
}

func TestSaveMemoryRejectsEmptyRequiredFields(t *testing.T) {
	tools := &Tools{Store: newFakeStore(), Embedder: &fakeEmbedder{vector: []float32{1}}}

	cases := []SaveMemoryArgs{
		{Type: "user", Slug: "", Description: "d", Body: "b", Global: true},
		{Type: "user", Slug: "s", Description: "", Body: "b", Global: true},
		{Type: "user", Slug: "s", Description: "d", Body: "", Global: true},
	}
	for _, args := range cases {
		if _, _, err := tools.SaveMemory(context.Background(), nil, args); err == nil {
			t.Errorf("SaveMemory(%+v) = nil error, want an error", args)
		}
	}
}

func TestSaveMemoryDuplicateSlugReturnsFriendlyError(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	args := SaveMemoryArgs{Type: "user", Slug: "dup", Description: "d", Body: "b", Global: true}

	if _, _, err := tools.SaveMemory(context.Background(), nil, args); err != nil {
		t.Fatalf("first SaveMemory: %v", err)
	}

	_, _, err := tools.SaveMemory(context.Background(), nil, args)
	if err == nil || !strings.Contains(err.Error(), "dup") || !strings.Contains(err.Error(), "update_memory") {
		t.Errorf("error = %v, want a message naming the slug and suggesting update_memory", err)
	}
}

func TestSaveMemoryWrapsEmbeddingFailure(t *testing.T) {
	embedFailure := errors.New("llama.cpp server returned 500")
	tools := &Tools{Store: newFakeStore(), Embedder: &fakeEmbedder{err: embedFailure}}

	_, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "user", Slug: "s", Description: "d", Body: "b", Global: true,
	})
	if !errors.Is(err, embedFailure) {
		t.Errorf("error = %v, want it to wrap %v", err, embedFailure)
	}
}

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

func TestSaveMemoryBlocksOnALikelyDuplicate(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	if _, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "user", Slug: "first", Description: "likes tabs", Body: "b", Global: true,
	}); err != nil {
		t.Fatalf("seeding: %v", err)
	}
	store.similarResp = &postgres.ScoredMemory{Memory: models.Memory{Slug: "first"}, Similarity: 0.95}

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
	store.similarResp = &postgres.ScoredMemory{Memory: models.Memory{Slug: "first"}, Similarity: 0.95}

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

func TestSaveMemoryAllowsOverridingTheGlobalItCollidedWith(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	if _, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "user", Slug: "global-tabs", Description: "likes tabs", Body: "b", Global: true,
	}); err != nil {
		t.Fatalf("seeding: %v", err)
	}
	store.similarResp = &postgres.ScoredMemory{Memory: models.Memory{Slug: "global-tabs"}, Similarity: 0.95}

	// Declaring the override is the acknowledgement the duplicate block asks for.
	_, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "user", Slug: "recall-spaces", Description: "spaces here", Body: "b", Project: "recall", Overrides: "global-tabs",
	})
	if err != nil {
		t.Fatalf("SaveMemory: %v", err)
	}
}
