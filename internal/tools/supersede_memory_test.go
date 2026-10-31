package tools

import (
	"context"
	"errors"
	"testing"

	"github.com/lonhutt/recall/internal/models"
)

func TestSupersedeMemorySucceeds(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	seedMemory(t, store, &fakeEmbedder{vector: []float32{1}}, tools)
	if _, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "reference", Slug: "replacement", Description: "d", Body: "b", Global: true,
	}); err != nil {
		t.Fatalf("seeding replacement: %v", err)
	}

	_, result, err := tools.SupersedeMemory(context.Background(), nil, SupersedeMemoryArgs{OldSlug: "seeded", NewSlug: "replacement"})
	if err != nil {
		t.Fatalf("SupersedeMemory: %v", err)
	}
	if result.Memory.SupersededBy != "replacement" {
		t.Errorf("SupersededBy = %q, want replacement", result.Memory.SupersededBy)
	}
}

func TestSupersedeMemoryRejectsMissingOldSlug(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	if _, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "reference", Slug: "replacement", Description: "d", Body: "b", Global: true,
	}); err != nil {
		t.Fatalf("seeding: %v", err)
	}

	_, _, err := tools.SupersedeMemory(context.Background(), nil, SupersedeMemoryArgs{OldSlug: "missing", NewSlug: "replacement"})
	if !errors.Is(err, models.ErrNotFound) {
		t.Errorf("err = %v, want it to wrap models.ErrNotFound", err)
	}
}

func TestSupersedeMemoryRejectsMissingNewSlug(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	seedMemory(t, store, &fakeEmbedder{vector: []float32{1}}, tools)

	_, _, err := tools.SupersedeMemory(context.Background(), nil, SupersedeMemoryArgs{OldSlug: "seeded", NewSlug: "missing"})
	if !errors.Is(err, models.ErrNotFound) {
		t.Errorf("err = %v, want it to wrap models.ErrNotFound", err)
	}
}

func TestSupersedeMemoryRejectsSelfSupersession(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	seedMemory(t, store, &fakeEmbedder{vector: []float32{1}}, tools)

	_, _, err := tools.SupersedeMemory(context.Background(), nil, SupersedeMemoryArgs{OldSlug: "seeded", NewSlug: "seeded"})
	if err == nil {
		t.Fatal("expected an error for old_slug == new_slug, got nil")
	}
}

func TestUpdateMemoryClearsSupersededBy(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}, MaxTokensPerChunk: 2048}
	seedMemory(t, store, &fakeEmbedder{vector: []float32{1}}, tools)
	if _, _, err := tools.SaveMemory(context.Background(), nil, SaveMemoryArgs{
		Type: "reference", Slug: "replacement", Description: "d", Body: "b", Global: true,
	}); err != nil {
		t.Fatalf("seeding replacement: %v", err)
	}
	if _, _, err := tools.SupersedeMemory(context.Background(), nil, SupersedeMemoryArgs{OldSlug: "seeded", NewSlug: "replacement"}); err != nil {
		t.Fatalf("SupersedeMemory: %v", err)
	}

	_, result, err := tools.UpdateMemory(context.Background(), nil, UpdateMemoryArgs{Slug: "seeded", ClearSupersededBy: true})
	if err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}
	if result.Memory.SupersededBy != "" {
		t.Errorf("SupersededBy = %q after clear_superseded_by, want empty", result.Memory.SupersededBy)
	}
}
