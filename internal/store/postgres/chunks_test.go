//go:build integration

package postgres_test

import (
	"context"
	"testing"

	"github.com/lonhutt/recall/internal/store/postgres"
)

func TestCreateMemoryPersistsChunks(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	_, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "chunked", Description: "d", Body: "b",
		Embedding: vector(1, 0), EmbeddingModel: "embeddinggemma",
		Chunks: []postgres.MemoryChunk{
			{ChunkIndex: 0, Text: "first chunk", Embedding: vector(1, 0)},
			{ChunkIndex: 1, Text: "second chunk", Embedding: vector(0, 1)},
		},
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	count, err := store.CountMemoryChunks(ctx, "chunked")
	if err != nil {
		t.Fatalf("CountMemoryChunks: %v", err)
	}
	if count != 2 {
		t.Errorf("chunk count = %d, want 2", count)
	}
}

func TestDeleteMemoryCascadesToChunks(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	_, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "cascaded", Description: "d", Body: "b",
		Embedding: vector(1, 0), EmbeddingModel: "embeddinggemma",
		Chunks: []postgres.MemoryChunk{{ChunkIndex: 0, Text: "only chunk", Embedding: vector(1, 0)}},
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	if err := store.DeleteMemory(ctx, "cascaded"); err != nil {
		t.Fatalf("DeleteMemory: %v", err)
	}

	// CountMemoryChunks joins through memories, so once the memory row is
	// gone the join returns zero rows, not an error.
	count, err := store.CountMemoryChunks(ctx, "cascaded")
	if err != nil {
		t.Fatalf("CountMemoryChunks: %v", err)
	}
	if count != 0 {
		t.Errorf("chunk count after delete = %d, want 0", count)
	}
}

func TestUpdateMemoryReplacesChunksWholesale(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	_, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "replaced", Description: "d", Body: "b",
		Embedding: vector(1, 0), EmbeddingModel: "embeddinggemma",
		Chunks: []postgres.MemoryChunk{
			{ChunkIndex: 0, Text: "old chunk one", Embedding: vector(1, 0)},
			{ChunkIndex: 1, Text: "old chunk two", Embedding: vector(0, 1)},
			{ChunkIndex: 2, Text: "old chunk three", Embedding: vector(1, 1)},
		},
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	newBody := "shorter body"
	newChunks := []postgres.MemoryChunk{{ChunkIndex: 0, Text: newBody, Embedding: vector(0, 1)}}
	if _, err := store.UpdateMemory(ctx, "replaced", postgres.MemoryPatch{Body: &newBody, Chunks: &newChunks}); err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}

	count, err := store.CountMemoryChunks(ctx, "replaced")
	if err != nil {
		t.Fatalf("CountMemoryChunks: %v", err)
	}
	if count != 1 {
		t.Errorf("chunk count after replace = %d, want 1 (old chunks 1-2 must not survive)", count)
	}
}

func TestUpdateMemoryLeavesChunksUntouchedWhenPatchOmitsThem(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	_, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "untouched", Description: "d", Body: "b",
		Embedding: vector(1, 0), EmbeddingModel: "embeddinggemma",
		Chunks: []postgres.MemoryChunk{{ChunkIndex: 0, Text: "chunk", Embedding: vector(1, 0)}},
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	newProject := "recall"
	if _, err := store.UpdateMemory(ctx, "untouched", postgres.MemoryPatch{Project: &newProject}); err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}

	count, err := store.CountMemoryChunks(ctx, "untouched")
	if err != nil {
		t.Fatalf("CountMemoryChunks: %v", err)
	}
	if count != 1 {
		t.Errorf("chunk count = %d, want 1 (a patch that doesn't touch Chunks must leave them alone)", count)
	}
}
