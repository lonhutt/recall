package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	pgvector "github.com/pgvector/pgvector-go"

	"github.com/lonhutt/recall/internal/models"
)

type ScoredMemory struct {
	models.Memory
	Score float64
}

// SearchMemories ranks memories by cosine similarity to embedding, most
// similar first, narrowed by f.
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
		var (
			m       models.Memory
			project sql.NullString
			agent   sql.NullString
			score   float64
		)
		err := rows.Scan(&m.ID, &m.Type, &m.Slug, &m.Description, &m.Body, &project, &agent,
			&m.Tags, &m.EmbeddingModel, &m.CreatedAt, &m.UpdatedAt, &score)
		if err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				break
			}
			return nil, err
		}
		m.Project = project.String
		m.Agent = agent.String
		out = append(out, ScoredMemory{Memory: m, Score: score})
	}
	return out, rows.Err()
}
