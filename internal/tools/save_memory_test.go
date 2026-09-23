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
	tools := &Tools{Store: store, Embedder: embedder, EmbeddingModel: "embeddinggemma", MaxTokensPerChunk: 2048}

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
