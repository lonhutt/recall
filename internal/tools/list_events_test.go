package tools

import (
	"context"
	"testing"
)

func TestListEventsReturnsLoggedEvents(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store}
	if _, _, err := tools.LogEvent(context.Background(), nil, LogEventArgs{Description: "deployed v1"}); err != nil {
		t.Fatalf("LogEvent: %v", err)
	}

	_, result, err := tools.ListEvents(context.Background(), nil, ListEventsArgs{})
	if err != nil {
		t.Fatalf("ListEvents: %v", err)
	}
	if len(result.Events) != 1 || result.Events[0].Description != "deployed v1" {
		t.Errorf("Events = %+v, want one deployed v1 event", result.Events)
	}
}

func TestListEventsRejectsInvalidOccurredAfter(t *testing.T) {
	tools := &Tools{Store: newFakeStore()}

	_, _, err := tools.ListEvents(context.Background(), nil, ListEventsArgs{OccurredAfter: "not-a-timestamp"})
	if err == nil {
		t.Fatal("expected an error for an invalid occurred_after, got nil")
	}
}

func TestListEventsRejectsInvalidOccurredBefore(t *testing.T) {
	tools := &Tools{Store: newFakeStore()}

	_, _, err := tools.ListEvents(context.Background(), nil, ListEventsArgs{OccurredBefore: "not-a-timestamp"})
	if err == nil {
		t.Fatal("expected an error for an invalid occurred_before, got nil")
	}
}
