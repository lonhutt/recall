package tools

import (
	"context"
	"testing"
)

func TestDeleteMemoryRemovesIt(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: store, Embedder: embedder}
	seedMemory(t, store, embedder, tools)

	_, result, err := tools.DeleteMemory(context.Background(), nil, DeleteMemoryArgs{Slug: "seeded"})
	if err != nil {
		t.Fatalf("DeleteMemory: %v", err)
	}
	if !result.Deleted {
		t.Errorf("Deleted = false, want true")
	}

	if _, err := store.GetMemoryBySlug(context.Background(), "seeded"); err == nil {
		t.Error("memory still present after delete")
	}
}

func TestDeleteMemoryNotFoundIsALoudError(t *testing.T) {
	tools := &Tools{Store: newFakeStore(), Embedder: &fakeEmbedder{vector: []float32{1}}}

	_, _, err := tools.DeleteMemory(context.Background(), nil, DeleteMemoryArgs{Slug: "typo-d-slug"})
	if err == nil {
		t.Fatal("expected an error for a missing slug, got nil (silent no-op)")
	}
}
