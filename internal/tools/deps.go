// Package tools implements Recall's MCP tool handlers.
package tools

import (
	"context"

	"github.com/lonhutt/recall/internal/embeddings"
	"github.com/lonhutt/recall/internal/models"
	"github.com/lonhutt/recall/internal/store/postgres"
)

// Store is the persistence surface the tool handlers need. It's satisfied by
// *postgres.Store; the interface exists only so handler tests can use a fake
// instead of a real database.
type Store interface {
	CreateMemory(ctx context.Context, m postgres.NewMemory) (*models.Memory, error)
	UpdateMemory(ctx context.Context, slug string, p postgres.MemoryPatch) (*models.Memory, error)
	GetMemoryBySlug(ctx context.Context, slug string) (*models.Memory, error)
	DeleteMemory(ctx context.Context, slug string) error
	ListMemories(ctx context.Context, f postgres.MemoryFilter, limit, offset int) ([]models.Memory, error)
	SearchMemories(ctx context.Context, embedding []float32, f postgres.MemoryFilter, limit int) ([]postgres.ScoredMemory, error)
	CreateLogEvent(ctx context.Context, e postgres.NewLogEvent) (*models.LogEvent, error)
	ListLogEvents(ctx context.Context, f postgres.EventFilter, limit, offset int) ([]models.LogEvent, error)
}

// Embedder generates text embeddings. Satisfied by *llamacpp.Client.
type Embedder interface {
	Embed(ctx context.Context, texts []string, kind embeddings.InputType) ([][]float32, error)
	CountTokens(ctx context.Context, text string) (int, error)
}

type Tools struct {
	Store          Store
	Embedder       Embedder
	EmbeddingModel string
	// MaxTokensPerChunk is the per-chunk token budget. <= 0 disables
	// chunking entirely, matching llamacpp.New's convention for the same
	// config value.
	MaxTokensPerChunk int
	// DocumentPrefix mirrors the prefix the Embedder prepends to every
	// document input. Chunking has to count tokens including it, or a chunk
	// packed right up to the budget would exceed it once Embed adds the
	// prefix and be rejected.
	DocumentPrefix string
}
