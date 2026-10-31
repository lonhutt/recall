//go:build integration

package postgres_test

import (
	"context"
	"errors"
	"strings"
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

func TestCreateMemoryLeavesRelationshipFieldsEmptyByDefault(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	mem, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "plain", Description: "d", Body: "b", Embedding: vector(1, 0),
	})
	if err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}
	if mem.SupersededBy != "" || mem.Overrides != "" {
		t.Errorf("SupersededBy = %q, Overrides = %q, want both empty for a memory that doesn't declare either", mem.SupersededBy, mem.Overrides)
	}

	got, err := store.GetMemoryBySlug(ctx, "plain")
	if err != nil {
		t.Fatalf("GetMemoryBySlug: %v", err)
	}
	if got.SupersededBy != "" || got.Overrides != "" {
		t.Errorf("re-fetched SupersededBy = %q, Overrides = %q, want both empty", got.SupersededBy, got.Overrides)
	}
}

func TestSupersedeMemorySetsSupersededBy(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "old-fact", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(old-fact): %v", err)
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "new-fact", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(new-fact): %v", err)
	}

	updated, err := store.SupersedeMemory(ctx, "old-fact", "new-fact")
	if err != nil {
		t.Fatalf("SupersedeMemory: %v", err)
	}
	if updated.SupersededBy != "new-fact" {
		t.Errorf("SupersededBy = %q, want new-fact", updated.SupersededBy)
	}

	got, err := store.GetMemoryBySlug(ctx, "old-fact")
	if err != nil {
		t.Fatalf("GetMemoryBySlug: %v", err)
	}
	if got.SupersededBy != "new-fact" {
		t.Errorf("re-fetched SupersededBy = %q, want new-fact", got.SupersededBy)
	}
}

func TestSupersedeMemoryReturnsNotFoundForEitherMissingSlug(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "exists", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory: %v", err)
	}

	if _, err := store.SupersedeMemory(ctx, "missing-old", "exists"); !errors.Is(err, models.ErrNotFound) || !strings.Contains(err.Error(), "missing-old") {
		t.Errorf("old-slug-missing: err = %v, want models.ErrNotFound naming missing-old", err)
	}
	if _, err := store.SupersedeMemory(ctx, "exists", "missing-new"); !errors.Is(err, models.ErrNotFound) || !strings.Contains(err.Error(), "missing-new") {
		t.Errorf("new-slug-missing: err = %v, want models.ErrNotFound naming missing-new", err)
	}
}

func TestUpdateMemoryClearsSupersededBy(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "old-fact-2", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(old-fact-2): %v", err)
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "new-fact-2", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(new-fact-2): %v", err)
	}
	if _, err := store.SupersedeMemory(ctx, "old-fact-2", "new-fact-2"); err != nil {
		t.Fatalf("SupersedeMemory: %v", err)
	}

	updated, err := store.UpdateMemory(ctx, "old-fact-2", postgres.MemoryPatch{ClearSupersededBy: true})
	if err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}
	if updated.SupersededBy != "" {
		t.Errorf("SupersededBy = %q after clearing, want empty", updated.SupersededBy)
	}
}

func TestUpdateMemoryWithoutClearSupersededByLeavesItAlone(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "old-fact-3", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(old-fact-3): %v", err)
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "new-fact-3", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(new-fact-3): %v", err)
	}
	if _, err := store.SupersedeMemory(ctx, "old-fact-3", "new-fact-3"); err != nil {
		t.Fatalf("SupersedeMemory: %v", err)
	}

	newProject := "recall"
	updated, err := store.UpdateMemory(ctx, "old-fact-3", postgres.MemoryPatch{Project: &newProject})
	if err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}
	if updated.SupersededBy != "new-fact-3" {
		t.Errorf("SupersededBy = %q, want it untouched by an unrelated patch", updated.SupersededBy)
	}
}

func TestCreateMemoryWithOverridesResolvesToId(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "global-pref", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(global-pref): %v", err)
	}

	newProject := "recall"
	mem, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "project", Slug: "project-pref", Description: "d", Body: "b", Embedding: vector(1, 0),
		Project: newProject, Overrides: "global-pref",
	})
	if err != nil {
		t.Fatalf("CreateMemory(project-pref): %v", err)
	}
	if mem.Overrides != "global-pref" {
		t.Errorf("Overrides = %q, want global-pref", mem.Overrides)
	}
}

func TestUpdateMemorySetsAndClearsOverrides(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "global-pref-2", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(global-pref-2): %v", err)
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "project", Slug: "project-pref-2", Description: "d", Body: "b", Embedding: vector(1, 0), Project: "recall",
	}); err != nil {
		t.Fatalf("CreateMemory(project-pref-2): %v", err)
	}

	overrides := "global-pref-2"
	updated, err := store.UpdateMemory(ctx, "project-pref-2", postgres.MemoryPatch{Overrides: &overrides})
	if err != nil {
		t.Fatalf("UpdateMemory(set): %v", err)
	}
	if updated.Overrides != "global-pref-2" {
		t.Errorf("Overrides = %q after set, want global-pref-2", updated.Overrides)
	}

	empty := ""
	cleared, err := store.UpdateMemory(ctx, "project-pref-2", postgres.MemoryPatch{Overrides: &empty})
	if err != nil {
		t.Fatalf("UpdateMemory(clear): %v", err)
	}
	if cleared.Overrides != "" {
		t.Errorf("Overrides = %q after clear, want empty", cleared.Overrides)
	}
}

func TestDeletingAnOverriddenOrSupersedingMemoryClearsThePointer(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "global-pref-3", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(global-pref-3): %v", err)
	}
	overrides := "global-pref-3"
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "project", Slug: "project-pref-3", Description: "d", Body: "b", Embedding: vector(1, 0),
		Project: "recall", Overrides: overrides,
	}); err != nil {
		t.Fatalf("CreateMemory(project-pref-3): %v", err)
	}

	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "old-fact-4", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(old-fact-4): %v", err)
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{
		Type: "reference", Slug: "new-fact-4", Description: "d", Body: "b", Embedding: vector(1, 0),
	}); err != nil {
		t.Fatalf("CreateMemory(new-fact-4): %v", err)
	}
	if _, err := store.SupersedeMemory(ctx, "old-fact-4", "new-fact-4"); err != nil {
		t.Fatalf("SupersedeMemory: %v", err)
	}

	// Delete the target of each relationship — the FK's ON DELETE SET NULL
	// (Task 1) must clear the referencing row's pointer, not error and not
	// leave it dangling.
	if err := store.DeleteMemory(ctx, "global-pref-3"); err != nil {
		t.Fatalf("DeleteMemory(global-pref-3): %v", err)
	}
	if err := store.DeleteMemory(ctx, "new-fact-4"); err != nil {
		t.Fatalf("DeleteMemory(new-fact-4): %v", err)
	}

	gotOverride, err := store.GetMemoryBySlug(ctx, "project-pref-3")
	if err != nil {
		t.Fatalf("GetMemoryBySlug(project-pref-3): %v", err)
	}
	if gotOverride.Overrides != "" {
		t.Errorf("Overrides = %q after its target was deleted, want empty (ON DELETE SET NULL)", gotOverride.Overrides)
	}

	gotSuperseded, err := store.GetMemoryBySlug(ctx, "old-fact-4")
	if err != nil {
		t.Fatalf("GetMemoryBySlug(old-fact-4): %v", err)
	}
	if gotSuperseded.SupersededBy != "" {
		t.Errorf("SupersededBy = %q after its target was deleted, want empty (ON DELETE SET NULL)", gotSuperseded.SupersededBy)
	}
}

func TestSupersedeMemoryLeavesUpdatedAtAlone(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	old, err := store.CreateMemory(ctx, postgres.NewMemory{Type: "reference", Slug: "stale", Description: "d", Body: "b", Embedding: vector(1, 0)})
	if err != nil {
		t.Fatalf("CreateMemory(stale): %v", err)
	}
	if _, err := store.CreateMemory(ctx, postgres.NewMemory{Type: "reference", Slug: "fresh", Description: "d", Body: "b", Embedding: vector(1, 0)}); err != nil {
		t.Fatalf("CreateMemory(fresh): %v", err)
	}

	updated, err := store.SupersedeMemory(ctx, "stale", "fresh")
	if err != nil {
		t.Fatalf("SupersedeMemory: %v", err)
	}
	if !updated.UpdatedAt.Equal(old.UpdatedAt) {
		t.Errorf("UpdatedAt = %v, want unchanged %v: superseding isn't an edit", updated.UpdatedAt, old.UpdatedAt)
	}
}
