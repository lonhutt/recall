//go:build integration

package postgres_test

import (
	"context"
	"errors"
	"testing"

	"github.com/lonhutt/recall/internal/models"
	"github.com/lonhutt/recall/internal/store/postgres"
)

// embeddingDimension mirrors the vector(768) width migration 000004 sets;
// the schema is the source of truth, this is just what a test row has to
// match to insert at all.
const embeddingDimension = 768

// vector returns an embeddingDimension-length vector with direction as its
// leading components and zeros elsewhere, so cosine-similarity comparisons
// between a handful of test vectors are predictable.
func vector(direction ...float32) []float32 {
	v := make([]float32, embeddingDimension)
	copy(v, direction)
	return v
}

func TestCreateAndGetMemory(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	created, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type:           "user",
		Slug:           "go-error-wrapping",
		Description:    "prefers %w",
		Body:           "wrap errors with %w so callers can errors.Is/As",
		Project:        "recall",
		Agent:          "claude-code",
		Tags:           []string{"go", "errors"},
		Embedding:      vector(1, 0),
		EmbeddingModel: "embeddinggemma",
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}
	if created.Slug != "go-error-wrapping" {
		t.Errorf("Slug = %q, want go-error-wrapping", created.Slug)
	}

	got, err := store.GetMemoryBySlug(ctx, "go-error-wrapping")
	if err != nil {
		t.Fatalf("GetMemoryBySlug: %v", err)
	}
	if got.Description != "prefers %w" || got.Project != "recall" || len(got.Tags) != 2 {
		t.Errorf("GetMemoryBySlug returned unexpected memory: %+v", got)
	}
}

func TestCreateMemoryDuplicateSlugReturnsError(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	m := postgres.NewMemory{Type: "reference", Slug: "dup-slug", Description: "d", Body: "b", Embedding: vector(1)}
	if _, err := store.CreateMemory(ctx, m); err != nil {
		t.Fatalf("first CreateMemory: %v", err)
	}

	_, err := store.CreateMemory(ctx, m)
	if !errors.Is(err, models.ErrDuplicateSlug) {
		t.Errorf("second CreateMemory error = %v, want ErrDuplicateSlug", err)
	}
}

func TestGetMemoryBySlugNotFound(t *testing.T) {
	store := newTestStore(t)

	_, err := store.GetMemoryBySlug(context.Background(), "does-not-exist")
	if !errors.Is(err, models.ErrNotFound) {
		t.Errorf("error = %v, want ErrNotFound", err)
	}
}

func TestUpdateMemoryPatchesOnlyGivenFields(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	_, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "project", Slug: "patch-me", Description: "old desc", Body: "old body",
		Project: "recall", Embedding: vector(1),
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	newDesc := "new desc"
	updated, err := store.UpdateMemory(ctx, "patch-me", postgres.MemoryPatch{Description: &newDesc})
	if err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}
	if updated.Description != "new desc" {
		t.Errorf("Description = %q, want %q", updated.Description, "new desc")
	}
	if updated.Body != "old body" {
		t.Errorf("Body = %q, want unchanged %q", updated.Body, "old body")
	}
	if updated.Project != "recall" {
		t.Errorf("Project = %q, want unchanged %q", updated.Project, "recall")
	}
}

func TestUpdateMemoryNotFound(t *testing.T) {
	store := newTestStore(t)
	newDesc := "x"

	_, err := store.UpdateMemory(context.Background(), "does-not-exist", postgres.MemoryPatch{Description: &newDesc})
	if !errors.Is(err, models.ErrNotFound) {
		t.Errorf("error = %v, want ErrNotFound", err)
	}
}

func TestDeleteMemory(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	_, err := store.CreateMemory(ctx, postgres.NewMemory{Type: "user", Slug: "delete-me", Description: "d", Body: "b", Embedding: vector(1)})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	if err := store.DeleteMemory(ctx, "delete-me"); err != nil {
		t.Fatalf("DeleteMemory: %v", err)
	}

	_, err = store.GetMemoryBySlug(ctx, "delete-me")
	if !errors.Is(err, models.ErrNotFound) {
		t.Errorf("memory still found after delete: %v", err)
	}
}

func TestDeleteMemoryNotFound(t *testing.T) {
	store := newTestStore(t)

	err := store.DeleteMemory(context.Background(), "does-not-exist")
	if !errors.Is(err, models.ErrNotFound) {
		t.Errorf("error = %v, want ErrNotFound", err)
	}
}

func TestListMemoriesFiltersByType(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	mustCreate := func(slug, typ string) {
		if _, err := store.CreateMemory(ctx, postgres.NewMemory{Type: typ, Slug: slug, Description: "d", Body: "b", Embedding: vector(1)}); err != nil {
			t.Fatalf("CreateMemory(%s): %v", slug, err)
		}
	}
	mustCreate("list-user-1", "user")
	mustCreate("list-feedback-1", "feedback")

	userType := "user"
	got, err := store.ListMemories(ctx, postgres.MemoryFilter{Type: &userType}, 10, 0)
	if err != nil {
		t.Fatalf("ListMemories: %v", err)
	}
	for _, m := range got {
		if m.Type != "user" {
			t.Errorf("ListMemories returned non-user memory: %+v", m)
		}
	}
	found := false
	for _, m := range got {
		if m.Slug == "list-user-1" {
			found = true
		}
	}
	if !found {
		t.Errorf("ListMemories did not include list-user-1: %+v", got)
	}
}
