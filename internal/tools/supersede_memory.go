package tools

import (
	"context"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type SupersedeMemoryArgs struct {
	OldSlug string `json:"old_slug" jsonschema:"slug of the memory that's no longer current"`
	NewSlug string `json:"new_slug" jsonschema:"slug of the memory that replaces it"`
}

type SupersedeMemoryResult struct {
	Memory MemoryDTO `json:"memory"`
}

// SupersedeMemory marks old_slug as superseded by new_slug. Undone with
// update_memory's clear_superseded_by.
func (t *Tools) SupersedeMemory(ctx context.Context, _ *mcp.CallToolRequest, a SupersedeMemoryArgs) (*mcp.CallToolResult, SupersedeMemoryResult, error) {
	if a.OldSlug == a.NewSlug {
		return nil, SupersedeMemoryResult{}, fmt.Errorf("old_slug and new_slug must be different (got %q for both)", a.OldSlug)
	}
	updated, err := t.Store.SupersedeMemory(ctx, a.OldSlug, a.NewSlug)
	if err != nil {
		return nil, SupersedeMemoryResult{}, err
	}
	return nil, SupersedeMemoryResult{Memory: newMemoryDTO(updated)}, nil
}
