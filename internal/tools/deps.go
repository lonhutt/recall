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
}

type Tools struct {
	Store          Store
	Embedder       Embedder
	EmbeddingModel string
}
