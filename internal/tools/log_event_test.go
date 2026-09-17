package tools

import (
	"context"
	"testing"
	"time"
)

func TestLogEventDefaultsOccurredAtToNow(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store}

	before := time.Now().UTC()
	_, result, err := tools.LogEvent(context.Background(), nil, LogEventArgs{Description: "deployed"})
	after := time.Now().UTC()
	if err != nil {
		t.Fatalf("LogEvent: %v", err)
	}

	occurred, err := time.Parse(time.RFC3339, result.Event.OccurredAt)
	if err != nil {
		t.Fatalf("OccurredAt %q is not RFC3339: %v", result.Event.OccurredAt, err)
	}
	if occurred.Before(before.Add(-time.Second)) || occurred.After(after.Add(time.Second)) {
		t.Errorf("OccurredAt = %v, want ~now (between %v and %v)", occurred, before, after)
	}
}

func TestLogEventUsesGivenOccurredAt(t *testing.T) {
	tools := &Tools{Store: newFakeStore()}

	_, result, err := tools.LogEvent(context.Background(), nil, LogEventArgs{
		Description: "deployed", OccurredAt: "2026-01-01T12:00:00Z",
	})
	if err != nil {
		t.Fatalf("LogEvent: %v", err)
	}
	if result.Event.OccurredAt != "2026-01-01T12:00:00Z" {
		t.Errorf("OccurredAt = %q, want 2026-01-01T12:00:00Z", result.Event.OccurredAt)
	}
}

func TestLogEventRejectsInvalidOccurredAt(t *testing.T) {
	tools := &Tools{Store: newFakeStore()}

	_, _, err := tools.LogEvent(context.Background(), nil, LogEventArgs{Description: "x", OccurredAt: "not-a-timestamp"})
	if err == nil {
		t.Fatal("expected an error for an invalid occurred_at, got nil")
	}
}

func TestLogEventRejectsEmptyDescription(t *testing.T) {
	tools := &Tools{Store: newFakeStore()}

	_, _, err := tools.LogEvent(context.Background(), nil, LogEventArgs{Description: "  "})
	if err == nil {
		t.Fatal("expected an error for an empty description, got nil")
	}
}

func TestLogEventDoesNotCallEmbedder(t *testing.T) {
	embedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: newFakeStore(), Embedder: embedder}

	if _, _, err := tools.LogEvent(context.Background(), nil, LogEventArgs{Description: "x"}); err != nil {
		t.Fatalf("LogEvent: %v", err)
	}
	if embedder.calls != 0 {
		t.Errorf("embedder called %d times, want 0 (log_event is a plain relational insert)", embedder.calls)
	}
}
