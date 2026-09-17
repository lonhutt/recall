package tools

import (
	"context"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type ListMemoriesArgs struct {
	Type    string   `json:"type,omitempty" jsonschema:"restrict to this memory type"`
	Project string   `json:"project,omitempty"`
	Agent   string   `json:"agent,omitempty"`
	Tags    []string `json:"tags,omitempty" jsonschema:"require all of these tags"`
	Limit   int      `json:"limit,omitempty" jsonschema:"max results, default 50, max 200"`
	Offset  int      `json:"offset,omitempty"`
}

type ListMemoriesResult struct {
	Memories []MemoryDTO `json:"memories"`
}

// ListMemories browses memory notes by metadata only; it never calls the
// embedder, so it's the cheap replacement for reading the old MEMORY.md index.
func (t *Tools) ListMemories(ctx context.Context, _ *mcp.CallToolRequest, a ListMemoriesArgs) (*mcp.CallToolResult, ListMemoriesResult, error) {
	limit := clamp(a.Limit, 1, 200, 50)
	filter := memoryFilterFrom(a.Type, a.Project, a.Agent, a.Tags)

	memories, err := t.Store.ListMemories(ctx, filter, limit, a.Offset)
	if err != nil {
		return nil, ListMemoriesResult{}, err
	}

	dtos := make([]MemoryDTO, len(memories))
	for i := range memories {
		dtos[i] = newMemoryDTO(&memories[i])
	}
	return nil, ListMemoriesResult{Memories: dtos}, nil
}
