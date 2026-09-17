//go:build integration

package postgres_test

import (
	"context"
	"testing"
	"time"

	"github.com/lonhutt/recall/internal/store/postgres"
)

func TestCreateAndListLogEvents(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	occurred := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	created, err := store.CreateLogEvent(ctx, postgres.NewLogEvent{
		OccurredAt:  occurred,
		Description: "deployed v1",
		Project:     "recall",
		Agent:       "claude-code",
		Tags:        []string{"deploy"},
		Metadata:    map[string]any{"version": "1.0.0"},
	})
	if err != nil {
		t.Fatalf("CreateLogEvent: %v", err)
	}
	if created.Description != "deployed v1" {
		t.Errorf("Description = %q, want %q", created.Description, "deployed v1")
	}
	if created.Metadata["version"] != "1.0.0" {
		t.Errorf("Metadata[version] = %v, want 1.0.0", created.Metadata["version"])
	}

	project := "recall"
	got, err := store.ListLogEvents(ctx, postgres.EventFilter{Project: &project}, 10, 0)
	if err != nil {
		t.Fatalf("ListLogEvents: %v", err)
	}
	if len(got) != 1 || got[0].Description != "deployed v1" {
		t.Errorf("ListLogEvents = %+v, want one deployed v1 event", got)
	}
}

func TestListLogEventsFiltersByOccurredAtRange(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	old := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	recent := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	if _, err := store.CreateLogEvent(ctx, postgres.NewLogEvent{OccurredAt: old, Description: "old event"}); err != nil {
		t.Fatalf("CreateLogEvent: %v", err)
	}
	if _, err := store.CreateLogEvent(ctx, postgres.NewLogEvent{OccurredAt: recent, Description: "recent event"}); err != nil {
		t.Fatalf("CreateLogEvent: %v", err)
	}

	cutoff := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	got, err := store.ListLogEvents(ctx, postgres.EventFilter{OccurredAfter: &cutoff}, 10, 0)
	if err != nil {
		t.Fatalf("ListLogEvents: %v", err)
	}
	if len(got) != 1 || got[0].Description != "recent event" {
		t.Errorf("ListLogEvents with OccurredAfter = %+v, want only recent event", got)
	}
}
