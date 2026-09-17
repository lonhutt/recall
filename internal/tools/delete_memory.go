package tools

import (
	"context"
	"errors"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/lonhutt/recall/internal/models"
)

type DeleteMemoryArgs struct {
	Slug string `json:"slug" jsonschema:"slug of the memory to delete"`
}

type DeleteMemoryResult struct {
	Deleted bool `json:"deleted"`
}

func (t *Tools) DeleteMemory(ctx context.Context, _ *mcp.CallToolRequest, a DeleteMemoryArgs) (*mcp.CallToolResult, DeleteMemoryResult, error) {
	if err := t.Store.DeleteMemory(ctx, a.Slug); err != nil {
		if errors.Is(err, models.ErrNotFound) {
			return nil, DeleteMemoryResult{}, fmt.Errorf("no memory found with slug %q", a.Slug)
		}
		return nil, DeleteMemoryResult{}, err
	}
	return nil, DeleteMemoryResult{Deleted: true}, nil
}
