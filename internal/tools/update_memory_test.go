package tools

import (
	"context"
	"errors"
	"testing"

	"github.com/lonhutt/recall/internal/embeddings"
)

func seedMemory(t *testing.T, store *fakeStore, embedder *fakeEmbedder, tools *Tools) {
	t.Helper()
	_, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "project", Slug: "seeded", Description: "old desc", Body: "old body",
	})
	if err != nil {
		t.Fatalf("seeding memory: %v", err)
	}
}

func TestUpdateMemoryPatchesOnlyGivenFields(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: store, Embedder: embedder, MaxTokensPerChunk: 2048}
	seedMemory(t, store, embedder, tools)

	newDesc := "new desc"
	_, result, err := tools.UpdateMemory(context.Background(), nil, UpdateMemoryArgs{Slug: "seeded", Description: &newDesc})
	if err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}
	if result.Memory.Description != "new desc" {
		t.Errorf("Description = %q, want %q", result.Memory.Description, "new desc")
	}
	if result.Memory.Body != "old body" {
		t.Errorf("Body = %q, want unchanged %q", result.Memory.Body, "old body")
	}
}

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

func TestUpdateMemorySkipsEmbeddingWhenNeitherDescriptionNorBodyChange(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: store, Embedder: embedder, MaxTokensPerChunk: 2048}
	seedMemory(t, store, embedder, tools)
	embedder.calls = 0

	newProject := "recall"
	_, _, err := tools.UpdateMemory(context.Background(), nil, UpdateMemoryArgs{Slug: "seeded", Project: &newProject})
	if err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}
	if embedder.calls != 0 {
		t.Errorf("embedder called %d times, want 0", embedder.calls)
	}
}

// A row's embedding_model stamp has to describe both of its vectors. If the
// configured model changed since the row was written, re-embedding only the
// field the caller touched would leave the other vector in the old model's
// space under the new model's name — exactly what search_memory's
// embedding_model filter exists to make impossible.
func TestUpdateMemoryReembedsBothFieldsWhenTheModelChanged(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: store, Embedder: embedder, EmbeddingModel: "old-model", MaxTokensPerChunk: 2048}
	seedMemory(t, store, embedder, tools)

	tools.EmbeddingModel = "new-model"
	embedder.calls = 0
	embedder.allCalls = nil
	store.lastChunks = nil

	newProject := "recall"
	_, _, err := tools.UpdateMemory(context.Background(), nil, UpdateMemoryArgs{Slug: "seeded", Project: &newProject})
	if err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}
	if embedder.calls != 2 {
		t.Fatalf("embedder called %d times, want 2 (description and body both re-embedded under the new model): %+v", embedder.calls, embedder.allCalls)
	}
	if len(embedder.allCalls[0].texts) != 1 || embedder.allCalls[0].texts[0] != "old desc" {
		t.Errorf("first embed call = %v, want the existing description re-embedded", embedder.allCalls[0].texts)
	}
	if len(embedder.allCalls[1].texts) != 1 || embedder.allCalls[1].texts[0] != "old body" {
		t.Errorf("second embed call = %v, want the existing body re-embedded", embedder.allCalls[1].texts)
	}
	if len(store.lastChunks) == 0 {
		t.Error("store got no chunks, want the body re-chunked under the new model")
	}
	if got := store.bySlug["seeded"].EmbeddingModel; got != "new-model" {
		t.Errorf("stored EmbeddingModel = %q, want %q", got, "new-model")
	}
}

func TestUpdateMemoryNotFoundReturnsFriendlyError(t *testing.T) {
	tools := &Tools{Store: newFakeStore(), Embedder: &fakeEmbedder{vector: []float32{1}}}
	newDesc := "x"

	_, _, err := tools.UpdateMemory(context.Background(), nil, UpdateMemoryArgs{Slug: "missing", Description: &newDesc})
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
}

func TestUpdateMemoryRejectsUnknownType(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: store, Embedder: embedder, MaxTokensPerChunk: 2048}
	seedMemory(t, store, embedder, tools)

	bogus := "bogus"
	_, _, err := tools.UpdateMemory(context.Background(), nil, UpdateMemoryArgs{Slug: "seeded", Type: &bogus})
	if err == nil {
		t.Fatal("expected an error for unknown type, got nil")
	}
}

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
