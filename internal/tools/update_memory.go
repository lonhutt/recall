package tools

import (
	"context"
	"errors"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/lonhutt/recall/internal/embeddings"
	"github.com/lonhutt/recall/internal/models"
	"github.com/lonhutt/recall/internal/store/postgres"
)

// UpdateMemoryArgs gives update_memory PATCH semantics: a nil field is left
// unchanged. Slug identifies the memory and is immutable.
type UpdateMemoryArgs struct {
	Slug        string    `json:"slug" jsonschema:"slug of the memory to update"`
	Type        *string   `json:"type,omitempty" jsonschema:"new memory type: user, feedback, project, or reference"`
	Description *string   `json:"description,omitempty"`
	Body        *string   `json:"body,omitempty"`
	Project     *string   `json:"project,omitempty"`
	Agent       *string   `json:"agent,omitempty"`
	Tags        *[]string `json:"tags,omitempty"`
}

type UpdateMemoryResult struct {
	Memory MemoryDTO `json:"memory"`
}

func (t *Tools) UpdateMemory(ctx context.Context, _ *mcp.CallToolRequest, a UpdateMemoryArgs) (*mcp.CallToolResult, UpdateMemoryResult, error) {
	if a.Type != nil && !models.IsValidMemoryType(*a.Type) {
		return nil, UpdateMemoryResult{}, fmt.Errorf("unknown memory type %q; expected one of %v", *a.Type, models.ValidMemoryTypes())
	}

	existing, err := t.Store.GetMemoryBySlug(ctx, a.Slug)
	if err != nil {
		if errors.Is(err, models.ErrNotFound) {
			return nil, UpdateMemoryResult{}, fmt.Errorf("no memory found with slug %q", a.Slug)
		}
		return nil, UpdateMemoryResult{}, err
	}

	patch := postgres.MemoryPatch{
		Type:        a.Type,
		Description: a.Description,
		Body:        a.Body,
		Project:     a.Project,
		Agent:       a.Agent,
		Tags:        a.Tags,
	}

	// Recall owns embedding-space consistency: any change to the text that
	// gets embedded re-embeds the full merged text, never leaving a stale vector.
	if a.Description != nil || a.Body != nil {
		desc, body := existing.Description, existing.Body
		if a.Description != nil {
			desc = *a.Description
		}
		if a.Body != nil {
			body = *a.Body
		}

		vectors, err := t.Embedder.Embed(ctx, []string{desc + "\n\n" + body}, embeddings.InputTypeDocument)
		if err != nil {
			return nil, UpdateMemoryResult{}, fmt.Errorf("generating embedding: %w", err)
		}
		patch.Embedding = vectors[0]
		patch.EmbeddingModel = &t.EmbeddingModel
	}

	updated, err := t.Store.UpdateMemory(ctx, a.Slug, patch)
	if err != nil {
		return nil, UpdateMemoryResult{}, err
	}
	return nil, UpdateMemoryResult{Memory: newMemoryDTO(updated)}, nil
}
