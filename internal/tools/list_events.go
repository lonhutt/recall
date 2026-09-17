package tools

import (
	"context"
	"fmt"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/lonhutt/recall/internal/store/postgres"
)

type ListEventsArgs struct {
	Project        string   `json:"project,omitempty"`
	Agent          string   `json:"agent,omitempty"`
	Tags           []string `json:"tags,omitempty" jsonschema:"require all of these tags"`
	OccurredAfter  string   `json:"occurred_after,omitempty" jsonschema:"RFC3339 timestamp lower bound"`
	OccurredBefore string   `json:"occurred_before,omitempty" jsonschema:"RFC3339 timestamp upper bound"`
	Limit          int      `json:"limit,omitempty" jsonschema:"max results, default 50, max 200"`
	Offset         int      `json:"offset,omitempty"`
}

type ListEventsResult struct {
	Events []LogEventDTO `json:"events"`
}

func (t *Tools) ListEvents(ctx context.Context, _ *mcp.CallToolRequest, a ListEventsArgs) (*mcp.CallToolResult, ListEventsResult, error) {
	limit := limitOrDefault(a.Limit, 1, 200, 50)
	filter := postgres.EventFilter{Tags: a.Tags}
	if a.Project != "" {
		filter.Project = &a.Project
	}
	if a.Agent != "" {
		filter.Agent = &a.Agent
	}
	if a.OccurredAfter != "" {
		ts, err := time.Parse(time.RFC3339, a.OccurredAfter)
		if err != nil {
			return nil, ListEventsResult{}, fmt.Errorf("occurred_after %q is not a valid RFC3339 timestamp: %w", a.OccurredAfter, err)
		}
		filter.OccurredAfter = &ts
	}
	if a.OccurredBefore != "" {
		ts, err := time.Parse(time.RFC3339, a.OccurredBefore)
		if err != nil {
			return nil, ListEventsResult{}, fmt.Errorf("occurred_before %q is not a valid RFC3339 timestamp: %w", a.OccurredBefore, err)
		}
		filter.OccurredBefore = &ts
	}

	events, err := t.Store.ListLogEvents(ctx, filter, limit, max(a.Offset, 0))
	if err != nil {
		return nil, ListEventsResult{}, err
	}

	dtos := make([]LogEventDTO, len(events))
	for i := range events {
		dtos[i] = newLogEventDTO(&events[i])
	}
	return nil, ListEventsResult{Events: dtos}, nil
}
