package postgres

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"time"

	pgvector "github.com/pgvector/pgvector-go"

	"github.com/lonhutt/recall/internal/models"
)

type ScoredMemory struct {
	models.Memory
	// Score is the fused rank value results are ordered by. It's only
	// comparable to other results from the same search.
	Score float64
	// Similarity is the cosine match against embedding: the better of the
	// description and best body chunk. Unlike Score it's absolute, so a weak
	// top result still looks weak.
	Similarity float64
	// OverriddenBy is set only when the search was scoped to a project and a
	// memory in that project overrides this one; the relationship only holds
	// within that project.
	OverriddenBy string
}

const (
	// rrfK is the usual reciprocal rank fusion constant; it keeps one leg's
	// top hit from swamping a memory both legs rank well.
	rrfK = 60
	// candidateDepth is how many candidates each leg contributes to fusion;
	// a leg can only promote what it fetched.
	candidateDepth = 50

	decayHalfLifeDurable  = 720 * 24 * time.Hour // user, reference
	decayHalfLifeVolatile = 90 * 24 * time.Hour  // project, feedback
)

// decayFactor is a (0,1] multiplier on a memory's fused score, halving every
// half-life of age since it was last updated.
func decayFactor(memType string, updatedAt time.Time) float64 {
	halfLife := decayHalfLifeVolatile
	if memType == string(models.MemoryTypeUser) || memType == string(models.MemoryTypeReference) {
		halfLife = decayHalfLifeDurable
	}
	return math.Exp(-math.Ln2 * time.Since(updatedAt).Hours() / halfLife.Hours())
}

// SearchMemories ranks memories against embedding and query, best first,
// narrowed by f; a Project filter also admits global memories. Superseded
// memories, and ones overridden within f.Project, rank below the rest;
// within each tier the order is a vector and full-text fusion, decayed by
// age. Callers should set f.EmbeddingModel; cosine distance across two
// models means nothing. An empty query skips the full-text leg.
func (s *Store) SearchMemories(ctx context.Context, embedding []float32, query string, f MemoryFilter, limit int) ([]ScoredMemory, error) {
	f.withGlobal = true
	vec := pgvector.NewVector(embedding)
	depth := max(limit, candidateDepth)

	descWhere, descFilterArgs := buildFilter(1, f)
	descArgs := append([]any{vec}, descFilterArgs...)
	descArgs = append(descArgs, depth)
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
	chunkArgs = append(chunkArgs, depth)
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

	// Best of description and best chunk, so a long memory surfaces on content
	// deep in its body without its chunks taking several slots.
	vectorScores := make(map[string]float64, len(descScores)+len(chunkScores))
	for id, score := range descScores {
		vectorScores[id] = score
	}
	for id, score := range chunkScores {
		if cur, ok := vectorScores[id]; !ok || score > cur {
			vectorScores[id] = score
		}
	}

	// The keyword leg catches exact slugs, identifiers and error text the
	// embedding ranks low.
	scored := []map[string]float64{vectorScores}
	if strings.TrimSpace(query) != "" {
		keywordScores, err := s.keywordScores(ctx, query, f, depth)
		if err != nil {
			return nil, fmt.Errorf("searching full text: %w", err)
		}
		scored = append(scored, keywordScores)
	}

	// Decay scales each leg's raw score before ranking, so it trades against
	// how good a match is. On the fused score it would trade against rank
	// alone, where neighbours differ by under 2% and a few days of age flip
	// them.
	if err := s.applyDecay(ctx, scored); err != nil {
		return nil, fmt.Errorf("applying recency decay: %w", err)
	}
	var legs [][]string
	for _, scores := range scored {
		legs = append(legs, rankedIDs(scores))
	}

	fused := map[string]float64{}
	for _, ids := range legs {
		for i, id := range ids {
			fused[id] += 1.0 / float64(rrfK+i+1)
		}
	}
	if len(fused) == 0 {
		return nil, nil
	}

	ids := make([]string, 0, len(fused))
	for id := range fused {
		ids = append(ids, id)
	}
	// Similarity is computed here rather than taken from the vector legs, which
	// only saw their top depth; a keyword-only row still gets its best chunk.
	rows, err := s.pool.Query(ctx, `
		SELECT `+memoryColumns+`, GREATEST(
			1 - (embedding <=> $2),
			(SELECT max(1 - (mc.embedding <=> $2)) FROM memory_chunks mc WHERE mc.memory_id = memories.id)
		)
		FROM memories WHERE id = ANY($1::uuid[])`, ids, vec)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []ScoredMemory
	for rows.Next() {
		var similarity float64
		m, err := scanMemory(rows, &similarity)
		if err != nil {
			return nil, err
		}
		out = append(out, ScoredMemory{Memory: *m, Score: fused[m.ID], Similarity: similarity})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	if f.Project != nil {
		overriddenBy, err := s.overriddenWithinProject(ctx, *f.Project)
		if err != nil {
			return nil, fmt.Errorf("resolving overrides within project: %w", err)
		}
		for i := range out {
			out[i].OverriddenBy = overriddenBy[out[i].ID]
		}
	}

	demoted := func(m ScoredMemory) bool { return m.SupersededBy != "" || m.OverriddenBy != "" }
	slices.SortFunc(out, func(a, b ScoredMemory) int {
		return cmp.Or(
			compareBool(demoted(a), demoted(b)),
			cmp.Compare(b.Score, a.Score),
			cmp.Compare(b.Similarity, a.Similarity),
			strings.Compare(a.ID, b.ID),
		)
	})
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

// applyDecay multiplies every leg score by its memory's decayFactor, in place.
func (s *Store) applyDecay(ctx context.Context, legs []map[string]float64) error {
	var ids []string
	for _, scores := range legs {
		for id := range scores {
			ids = append(ids, id)
		}
	}
	if len(ids) == 0 {
		return nil
	}
	rows, err := s.pool.Query(ctx, `SELECT id, type, updated_at FROM memories WHERE id = ANY($1::uuid[])`, ids)
	if err != nil {
		return err
	}
	defer rows.Close()
	factor := map[string]float64{}
	for rows.Next() {
		var id, memType string
		var updatedAt time.Time
		if err := rows.Scan(&id, &memType, &updatedAt); err != nil {
			return err
		}
		factor[id] = decayFactor(memType, updatedAt)
	}
	if err := rows.Err(); err != nil {
		return err
	}
	for _, scores := range legs {
		for id := range scores {
			scores[id] *= factor[id]
		}
	}
	return nil
}

// compareBool orders false before true.
func compareBool(a, b bool) int {
	switch {
	case a == b:
		return 0
	case a:
		return 1
	default:
		return -1
	}
}

// overriddenWithinProject maps a memory's id to the slug of the memory in
// project that overrides it.
func (s *Store) overriddenWithinProject(ctx context.Context, project string) (map[string]string, error) {
	rows, err := s.pool.Query(ctx, `SELECT overrides, slug FROM memories WHERE project = $1 AND overrides IS NOT NULL AND superseded_by IS NULL ORDER BY slug`, project)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := map[string]string{}
	for rows.Next() {
		var targetID, slug string
		if err := rows.Scan(&targetID, &slug); err != nil {
			return nil, err
		}
		out[targetID] = slug
	}
	return out, rows.Err()
}

// keywordScores runs the full-text leg. plainto_tsquery would AND every word,
// which makes a natural-language query match almost nothing, so the query's
// lexemes are ORed instead and ts_rank rewards memories matching more of
// them. They're quoted one by one rather than rewriting plainto_tsquery's
// text, since a lexeme can contain & itself (a URL with a query string). A
// query with no lexemes (all stopwords, only punctuation) matches no rows.
func (s *Store) keywordScores(ctx context.Context, query string, f MemoryFilter, depth int) (map[string]float64, error) {
	where, filterArgs := buildFilter(1, f)
	cond := "search_tsv @@ kq.q"
	if where != "" {
		cond += " AND " + strings.TrimPrefix(where, "WHERE ")
	}
	args := append([]any{query}, filterArgs...)
	args = append(args, depth)
	q := fmt.Sprintf(`
		SELECT id, ts_rank(search_tsv, kq.q) AS score
		FROM memories,
			(SELECT string_agg('''' || replace(replace(l, '\', '\\'), '''', '''''') || '''', ' | ')::tsquery AS q
			 FROM unnest(tsvector_to_array(to_tsvector('english', $1))) l) kq
		WHERE %s
		ORDER BY score DESC, id
		LIMIT $%d
	`, cond, len(args))
	return s.scoreByID(ctx, q, args)
}

// rankedIDs orders scores best first, breaking ties by id so two identical
// searches always fuse the same ranks.
func rankedIDs(scores map[string]float64) []string {
	ids := make([]string, 0, len(scores))
	for id := range scores {
		ids = append(ids, id)
	}
	slices.SortFunc(ids, func(a, b string) int {
		return cmp.Or(cmp.Compare(scores[b], scores[a]), strings.Compare(a, b))
	})
	return ids
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

// FindSimilarMemory is save_memory's duplicate check: the closest current
// memory of memType by description embedding, at or above threshold, or nil.
// It looks in project first and falls back to global memories.
func (s *Store) FindSimilarMemory(ctx context.Context, embedding []float32, embeddingModel, memType, project string, threshold float64) (*ScoredMemory, error) {
	vec := pgvector.NewVector(embedding)

	tryScope := func(scope string, args ...any) (*ScoredMemory, error) {
		query := fmt.Sprintf(`
			SELECT %s, 1 - (embedding <=> $1)
			FROM memories
			WHERE embedding_model = $2 AND type = $3 AND superseded_by IS NULL AND %s
			ORDER BY embedding <=> $1
			LIMIT 1
		`, memoryColumns, scope)
		var similarity float64
		m, err := scanMemory(s.pool.QueryRow(ctx, query, append([]any{vec, embeddingModel, memType}, args...)...), &similarity)
		if errors.Is(err, models.ErrNotFound) || (err == nil && similarity < threshold) {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}
		return &ScoredMemory{Memory: *m, Similarity: similarity}, nil
	}

	if project != "" {
		hit, err := tryScope("project = $4", project)
		if err != nil || hit != nil {
			return hit, err
		}
	}
	return tryScope("project IS NULL")
}
