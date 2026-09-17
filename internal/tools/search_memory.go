package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/lonhutt/recall/internal/embeddings"
)

type SearchMemoryArgs struct {
	Query   string   `json:"query" jsonschema:"natural-language search text"`
	Type    string   `json:"type,omitempty" jsonschema:"restrict to this memory type"`
	Project string   `json:"project,omitempty"`
	Agent   string   `json:"agent,omitempty"`
	Tags    []string `json:"tags,omitempty" jsonschema:"require all of these tags"`
	Limit   int      `json:"limit,omitempty" jsonschema:"max results, default 10, max 50"`
}

type SearchMemoryResult struct {
	Results []ScoredMemoryDTO `json:"results"`
}

func (t *Tools) SearchMemory(ctx context.Context, _ *mcp.CallToolRequest, a SearchMemoryArgs) (*mcp.CallToolResult, SearchMemoryResult, error) {
	if strings.TrimSpace(a.Query) == "" {
		return nil, SearchMemoryResult{}, fmt.Errorf("query is required")
	}
	limit := limitOrDefault(a.Limit, 1, 50, 10)

	vectors, err := t.Embedder.Embed(ctx, []string{a.Query}, embeddings.InputTypeQuery)
	if err != nil {
		return nil, SearchMemoryResult{}, fmt.Errorf("generating embedding: %w", err)
	}

	filter := memoryFilterFrom(a.Type, a.Project, a.Agent, a.Tags)
	// scope the search to rows the current model embedded; a distance against
	// a vector from another model doesn't mean anything, so a half
	// re-embedded table would quietly degrade ranking instead of just
	// returning fewer rows.
	filter.EmbeddingModel = &t.EmbeddingModel
	scored, err := t.Store.SearchMemories(ctx, vectors[0], filter, limit)
	if err != nil {
		return nil, SearchMemoryResult{}, err
	}

	results := make([]ScoredMemoryDTO, len(scored))
	for i, s := range scored {
		results[i] = ScoredMemoryDTO{MemoryDTO: newMemoryDTO(&s.Memory), Score: s.Score}
	}
	return nil, SearchMemoryResult{Results: results}, nil
}
