package postgres

import (
	"context"
	"fmt"

	pgvector "github.com/pgvector/pgvector-go"

	"github.com/lonhutt/recall/internal/models"
)

type ScoredMemory struct {
	models.Memory
	Score float64
}

// SearchMemories ranks memories by cosine similarity to embedding, most
// similar first, narrowed by f. Callers should set f.EmbeddingModel; the
// cosine distance between vectors from two different models doesn't mean
// anything.
func (s *Store) SearchMemories(ctx context.Context, embedding []float32, f MemoryFilter, limit int) ([]ScoredMemory, error) {
	where, filterArgs := buildFilter(1, f)
	args := append([]any{pgvector.NewVector(embedding)}, filterArgs...)
	args = append(args, limit)

	query := fmt.Sprintf(`
		SELECT %s, 1 - (embedding <=> $1) AS score
		FROM memories
		%s
		ORDER BY embedding <=> $1
		LIMIT $%d
	`, memoryColumns, where, len(args))

	rows, err := s.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []ScoredMemory
	for rows.Next() {
		var score float64
		m, err := scanMemory(rows, &score)
		if err != nil {
			return nil, err
		}
		out = append(out, ScoredMemory{Memory: *m, Score: score})
	}
	return out, rows.Err()
}
