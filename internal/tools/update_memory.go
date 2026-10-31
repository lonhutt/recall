package tools

import (
	"context"
	"errors"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"

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
	Overrides   *string   `json:"overrides,omitempty" jsonschema:"slug of a global memory to override, or empty string to clear"`
	// Only clearing is offered; supersede_memory is the one way to set it.
	ClearSupersededBy bool `json:"clear_superseded_by,omitempty" jsonschema:"true un-marks this memory as superseded"`
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

	// Checked against where the memory ends up, so moving it to global while
	// it still overrides something fails too.
	project, overrides := existing.Project, existing.Overrides
	if a.Project != nil {
		project = *a.Project
	}
	if a.Overrides != nil {
		overrides = *a.Overrides
	}
	if a.Project != nil || a.Overrides != nil {
		if err := t.validateOverrides(ctx, project, overrides); err != nil {
			return nil, UpdateMemoryResult{}, err
		}
	}

	patch := postgres.MemoryPatch{
		Type:              a.Type,
		Description:       a.Description,
		Body:              a.Body,
		Project:           a.Project,
		Agent:             a.Agent,
		Tags:              a.Tags,
		Overrides:         a.Overrides,
		ClearSupersededBy: a.ClearSupersededBy,
	}

	// Description and body embed independently: each is only re-embedded
	// when the caller actually changed it, not both whenever either changes.
	//
	// The exception is a configured embedding model that no longer matches
	// the one this row was embedded with. Then both fields are re-embedded
	// even if the caller touched one or neither — otherwise the row would
	// end up stamped with the new model name while one of its two vectors
	// is still from the old model, and search_memory's embedding_model
	// filter exists precisely to prevent ever comparing vectors across
	// models.
	modelChanged := existing.EmbeddingModel != t.EmbeddingModel
	reembedDescription := a.Description != nil || modelChanged
	reembedBody := a.Body != nil || modelChanged

	if reembedDescription || reembedBody {
		patch.EmbeddingModel = &t.EmbeddingModel
	}
	if reembedDescription {
		desc := existing.Description
		if a.Description != nil {
			desc = *a.Description
		}
		vec, err := t.embedDescription(ctx, desc)
		if err != nil {
			return nil, UpdateMemoryResult{}, fmt.Errorf("generating description embedding: %w", err)
		}
		patch.Embedding = vec
	}
	if reembedBody {
		body := existing.Body
		if a.Body != nil {
			body = *a.Body
		}
		chunks, err := t.chunkAndEmbedBody(ctx, body)
		if err != nil {
			return nil, UpdateMemoryResult{}, fmt.Errorf("generating body chunk embeddings: %w", err)
		}
		patch.Chunks = &chunks
	}

	updated, err := t.Store.UpdateMemory(ctx, a.Slug, patch)
	if err != nil {
		return nil, UpdateMemoryResult{}, err
	}
	return nil, UpdateMemoryResult{Memory: newMemoryDTO(updated)}, nil
}
