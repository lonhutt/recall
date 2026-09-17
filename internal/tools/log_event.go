package tools

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/lonhutt/recall/internal/store/postgres"
)

type LogEventArgs struct {
	Description string         `json:"description" jsonschema:"what happened"`
	OccurredAt  string         `json:"occurred_at,omitempty" jsonschema:"RFC3339 timestamp; defaults to now"`
	Project     string         `json:"project,omitempty"`
	Agent       string         `json:"agent,omitempty"`
	Tags        []string       `json:"tags,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

type LogEventResult struct {
	Event LogEventDTO `json:"event"`
}

// LogEvent is a plain relational insert; it never calls the embedder because
// episodic entries aren't semantically searched in v1.
func (t *Tools) LogEvent(ctx context.Context, _ *mcp.CallToolRequest, a LogEventArgs) (*mcp.CallToolResult, LogEventResult, error) {
	if strings.TrimSpace(a.Description) == "" {
		return nil, LogEventResult{}, fmt.Errorf("description is required")
	}

	occurredAt := time.Now().UTC()
	if a.OccurredAt != "" {
		parsed, err := time.Parse(time.RFC3339, a.OccurredAt)
		if err != nil {
			return nil, LogEventResult{}, fmt.Errorf("occurred_at %q is not a valid RFC3339 timestamp: %w", a.OccurredAt, err)
		}
		occurredAt = parsed
	}

	event, err := t.Store.CreateLogEvent(ctx, postgres.NewLogEvent{
		OccurredAt:  occurredAt,
		Description: a.Description,
		Project:     a.Project,
		Agent:       a.Agent,
		Tags:        a.Tags,
		Metadata:    a.Metadata,
	})
	if err != nil {
		return nil, LogEventResult{}, err
	}
	return nil, LogEventResult{Event: newLogEventDTO(event)}, nil
}
