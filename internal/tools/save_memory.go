package tools

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/lonhutt/recall/internal/models"
	"github.com/lonhutt/recall/internal/store/postgres"
)

type SaveMemoryArgs struct {
	Type        string   `json:"type" jsonschema:"memory type: user, feedback, project, or reference"`
	Slug        string   `json:"slug" jsonschema:"short unique url-safe identifier, e.g. go-error-wrapping"`
	Description string   `json:"description" jsonschema:"one-line summary of this memory"`
	Body        string   `json:"body" jsonschema:"full memory content; may use [[slug]] links to other memories"`
	Project     string   `json:"project,omitempty" jsonschema:"project this memory relates to, if any"`
	Agent       string   `json:"agent,omitempty" jsonschema:"agent/client saving this, e.g. claude-code"`
	Tags        []string `json:"tags,omitempty" jsonschema:"arbitrary labels for filtering"`
}

type SaveMemoryResult struct {
	Memory MemoryDTO `json:"memory"`
}

func (t *Tools) SaveMemory(ctx context.Context, _ *mcp.CallToolRequest, a SaveMemoryArgs) (*mcp.CallToolResult, SaveMemoryResult, error) {
	if !models.IsValidMemoryType(a.Type) {
		return nil, SaveMemoryResult{}, fmt.Errorf("unknown memory type %q; expected one of %v", a.Type, models.ValidMemoryTypes())
	}
	if strings.TrimSpace(a.Slug) == "" {
		return nil, SaveMemoryResult{}, fmt.Errorf("slug is required")
	}
	if strings.TrimSpace(a.Description) == "" {
		return nil, SaveMemoryResult{}, fmt.Errorf("description is required")
	}
	if strings.TrimSpace(a.Body) == "" {
		return nil, SaveMemoryResult{}, fmt.Errorf("body is required")
	}

	descVec, err := t.embedDescription(ctx, a.Description)
	if err != nil {
		return nil, SaveMemoryResult{}, fmt.Errorf("generating description embedding: %w", err)
	}
	chunks, err := t.chunkAndEmbedBody(ctx, a.Body)
	if err != nil {
		return nil, SaveMemoryResult{}, fmt.Errorf("generating body chunk embeddings: %w", err)
	}

	mem, err := t.Store.CreateMemory(ctx, postgres.NewMemory{
		Type:           a.Type,
		Slug:           a.Slug,
		Description:    a.Description,
		Body:           a.Body,
		Project:        a.Project,
		Agent:          a.Agent,
		Tags:           a.Tags,
		Embedding:      descVec,
		EmbeddingModel: t.EmbeddingModel,
		Chunks:         chunks,
	})
	if err != nil {
		if errors.Is(err, models.ErrDuplicateSlug) {
			return nil, SaveMemoryResult{}, fmt.Errorf("a memory with slug %q already exists; use update_memory to modify it", a.Slug)
		}
		return nil, SaveMemoryResult{}, err
	}

	return nil, SaveMemoryResult{Memory: newMemoryDTO(mem)}, nil
}
