package postgres

import (
	"context"
	"fmt"
	"sort"

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
//
// Ranking is two-phase: a memory's description embedding (memories.embedding)
// is the primary signal, since it's always small and reflects the whole
// memory; a memory's body-chunk embeddings (memory_chunks.embedding) are a
// secondary signal that surfaces a long memory when a query only matches
// content deep in its body. A memory's final score is the better of the
// two, so one long memory's several chunks can't crowd out other memories
// by occupying multiple slots in the ranking.
func (s *Store) SearchMemories(ctx context.Context, embedding []float32, f MemoryFilter, limit int) ([]ScoredMemory, error) {
	vec := pgvector.NewVector(embedding)

	descWhere, descFilterArgs := buildFilter(1, f)
	descArgs := append([]any{vec}, descFilterArgs...)
	descArgs = append(descArgs, limit)
	descQuery := fmt.Sprintf(`
		SELECT id, 1 - (embedding <=> $1) AS score
		FROM memories
		%s
		ORDER BY embedding <=> $1
		LIMIT $%d
	`, descWhere, len(descArgs))

	descScores, err := s.scoreByID(ctx, descQuery, descArgs)
	if err != nil {
		return nil, fmt.Errorf("searching description embeddings: %w", err)
	}

	chunkWhere, chunkFilterArgs := buildFilter(1, f)
	chunkArgs := append([]any{vec}, chunkFilterArgs...)
	chunkArgs = append(chunkArgs, limit)
	chunkQuery := fmt.Sprintf(`
		SELECT memory_id, score FROM (
			SELECT DISTINCT ON (mc.memory_id) mc.memory_id, 1 - (mc.embedding <=> $1) AS score
			FROM memory_chunks mc
			JOIN memories m ON m.id = mc.memory_id
			%s
			ORDER BY mc.memory_id, mc.embedding <=> $1
		) best
		ORDER BY score DESC
		LIMIT $%d
	`, chunkWhere, len(chunkArgs))

	chunkScores, err := s.scoreByID(ctx, chunkQuery, chunkArgs)
	if err != nil {
		return nil, fmt.Errorf("searching body chunk embeddings: %w", err)
	}

	best := make(map[string]float64, len(descScores)+len(chunkScores))
	for id, score := range descScores {
		best[id] = score
	}
	for id, score := range chunkScores {
		if cur, ok := best[id]; !ok || score > cur {
			best[id] = score
		}
	}
	if len(best) == 0 {
		return nil, nil
	}

	ids := make([]string, 0, len(best))
	for id := range best {
		ids = append(ids, id)
	}
	rows, err := s.pool.Query(ctx, `SELECT `+memoryColumns+` FROM memories WHERE id = ANY($1::uuid[])`, ids)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []ScoredMemory
	for rows.Next() {
		m, err := scanMemory(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, ScoredMemory{Memory: *m, Score: best[m.ID]})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	// stable so two identical searches can't reorder equally-scored
	// memories relative to each other.
	sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

// scoreByID runs query (which must select exactly (id uuid, score float8)
// per row) and returns the results as a map.
func (s *Store) scoreByID(ctx context.Context, query string, args []any) (map[string]float64, error) {
	rows, err := s.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	scores := map[string]float64{}
	for rows.Next() {
		var id string
		var score float64
		if err := rows.Scan(&id, &score); err != nil {
			return nil, err
		}
		scores[id] = score
	}
	return scores, rows.Err()
}
