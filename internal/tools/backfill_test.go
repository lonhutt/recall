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
