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

// duplicateSimilarityThreshold is hardcoded until real usage says what it
// should be.
const duplicateSimilarityThreshold = 0.92

type SaveMemoryArgs struct {
	Type        string   `json:"type" jsonschema:"memory type: user, feedback, project, or reference"`
	Slug        string   `json:"slug" jsonschema:"short unique url-safe identifier, e.g. go-error-wrapping"`
	Description string   `json:"description" jsonschema:"one-line summary of this memory"`
	Body        string   `json:"body" jsonschema:"full memory content; may use [[slug]] links to other memories"`
	Project     string   `json:"project,omitempty" jsonschema:"project this memory relates to; required unless global is true"`
	Agent       string   `json:"agent,omitempty" jsonschema:"agent/client saving this, e.g. claude-code"`
	Tags        []string `json:"tags,omitempty" jsonschema:"arbitrary labels for filtering"`
	Global      bool     `json:"global,omitempty" jsonschema:"save without a project scope; required if project is omitted"`
	Overrides   string   `json:"overrides,omitempty" jsonschema:"slug of a global memory this project-scoped memory takes precedence over"`
	Force       bool     `json:"force,omitempty" jsonschema:"bypass the near-duplicate check and save anyway"`
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
	// No silent global saves: an agent that forgot project would otherwise
	// leak a project-specific memory into every other project's searches.
	if a.Project == "" && !a.Global {
		return nil, SaveMemoryResult{}, fmt.Errorf("project is required unless global is explicitly set to true")
	}
	if err := t.validateOverrides(ctx, a.Project, a.Overrides); err != nil {
		return nil, SaveMemoryResult{}, err
	}

	descVec, err := t.embedDescription(ctx, a.Description)
	if err != nil {
		return nil, SaveMemoryResult{}, fmt.Errorf("generating description embedding: %w", err)
	}
	// Before chunking the body, so a blocked save doesn't spend embed calls.
	if !a.Force {
		similar, err := t.Store.FindSimilarMemory(ctx, descVec, t.EmbeddingModel, a.Type, a.Project, duplicateSimilarityThreshold)
		if err != nil {
			return nil, SaveMemoryResult{}, fmt.Errorf("checking for similar memories: %w", err)
		}
		// Overriding the very memory it collided with is the answer the block
		// asks for, not another duplicate.
		if similar != nil && similar.Slug != a.Overrides {
			scope := "global"
			if similar.Project != "" {
				scope = fmt.Sprintf("project %q", similar.Project)
			}
			return nil, SaveMemoryResult{}, fmt.Errorf(
				"a similar memory already exists: %q (%s, similarity %.2f); use update_memory to modify it, or pass force: true to save anyway",
				similar.Slug, scope, similar.Similarity,
			)
		}
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
		Overrides:      a.Overrides,
	})
	if err != nil {
		if errors.Is(err, models.ErrDuplicateSlug) {
			return nil, SaveMemoryResult{}, fmt.Errorf("a memory with slug %q already exists; use update_memory to modify it", a.Slug)
		}
		return nil, SaveMemoryResult{}, err
	}

	return nil, SaveMemoryResult{Memory: newMemoryDTO(mem)}, nil
}
