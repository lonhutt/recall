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
	tools := &Tools{Store: store, Embedder: embedder}
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

func TestUpdateMemoryReembedsOnBodyOrDescriptionChange(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: store, Embedder: embedder}
	seedMemory(t, store, embedder, tools)
	embedder.calls = 0 // reset after the seed's own save_memory embed call

	newBody := "new body"
	_, _, err := tools.UpdateMemory(context.Background(), nil, UpdateMemoryArgs{Slug: "seeded", Body: &newBody})
	if err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}
	if embedder.calls != 1 {
		t.Errorf("embedder called %d times, want 1", embedder.calls)
	}
	if embedder.lastKind != embeddings.InputTypeDocument {
		t.Errorf("embedder kind = %q, want document", embedder.lastKind)
	}
	// Must re-embed the merged text (unchanged description + new body), not just the changed field.
	if len(embedder.lastTexts) != 1 || embedder.lastTexts[0] != "old desc\n\nnew body" {
		t.Errorf("embedder texts = %v, want merged text", embedder.lastTexts)
	}
}

func TestUpdateMemorySkipsEmbeddingWhenNeitherDescriptionNorBodyChange(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: store, Embedder: embedder}
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
	tools := &Tools{Store: store, Embedder: embedder}
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
	tools := &Tools{Store: store, Embedder: seedEmbedder}
	seedMemory(t, store, seedEmbedder, tools)

	embedFailure := errors.New("llama.cpp server returned 500")
	tools.Embedder = &fakeEmbedder{err: embedFailure}

	newBody := "b2"
	_, _, err := tools.UpdateMemory(context.Background(), nil, UpdateMemoryArgs{Slug: "seeded", Body: &newBody})
	if !errors.Is(err, embedFailure) {
		t.Errorf("error = %v, want it to wrap %v", err, embedFailure)
	}
}
