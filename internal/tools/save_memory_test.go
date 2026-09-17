package tools

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/lonhutt/recall/internal/embeddings"
)

func TestSaveMemorySucceeds(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1, 2, 3}}
	tools := &Tools{Store: store, Embedder: embedder, EmbeddingModel: "embeddinggemma"}

	_, result, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "user", Slug: "likes-go", Description: "likes Go", Body: "prefers Go for backend work",
	})
	if err != nil {
		t.Fatalf("SaveMemory returned error: %v", err)
	}
	if result.Memory.Slug != "likes-go" {
		t.Errorf("Memory.Slug = %q, want likes-go", result.Memory.Slug)
	}
	if embedder.calls != 1 {
		t.Errorf("embedder called %d times, want 1", embedder.calls)
	}
	if embedder.lastKind != embeddings.InputTypeDocument {
		t.Errorf("embedder called with kind %q, want document", embedder.lastKind)
	}
}

func TestSaveMemoryRejectsUnknownType(t *testing.T) {
	tools := &Tools{Store: newFakeStore(), Embedder: &fakeEmbedder{vector: []float32{1}}}

	_, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "bogus", Slug: "s", Description: "d", Body: "b",
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
		{Type: "user", Slug: "", Description: "d", Body: "b"},
		{Type: "user", Slug: "s", Description: "", Body: "b"},
		{Type: "user", Slug: "s", Description: "d", Body: ""},
	}
	for _, args := range cases {
		if _, _, err := tools.SaveMemory(context.Background(), nil, args); err == nil {
			t.Errorf("SaveMemory(%+v) = nil error, want an error", args)
		}
	}
}

func TestSaveMemoryDuplicateSlugReturnsFriendlyError(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}}
	args := SaveMemoryArgs{Type: "user", Slug: "dup", Description: "d", Body: "b"}

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
		Type: "user", Slug: "s", Description: "d", Body: "b",
	})
	if !errors.Is(err, embedFailure) {
		t.Errorf("error = %v, want it to wrap %v", err, embedFailure)
	}
}
