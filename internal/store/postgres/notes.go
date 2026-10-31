package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	pgvector "github.com/pgvector/pgvector-go"

	"github.com/lonhutt/recall/internal/models"
)

type NewMemory struct {
	Type           string
	Slug           string
	Description    string
	Body           string
	Project        string
	Agent          string
	Tags           []string
	Embedding      []float32
	EmbeddingModel string
	Chunks         []MemoryChunk
	// Overrides is the slug of a global memory this one takes precedence over.
	Overrides string
}

// MemoryPatch gives update_memory PATCH semantics: a nil field is left
// unchanged. Description/Body use *string rather than string so "unchanged"
// and "clear to empty" are distinguishable.
type MemoryPatch struct {
	Type           *string
	Description    *string
	Body           *string
	Project        *string
	Agent          *string
	Tags           *[]string
	Embedding      []float32
	EmbeddingModel *string
	Chunks         *[]MemoryChunk
	// ClearSupersededBy is the only way to unset superseded_by here; setting
	// it goes through SupersedeMemory.
	ClearSupersededBy bool
	// Overrides sets the overrides slug; "" clears it.
	Overrides *string
}

const memoryColumns = `id, type, slug, description, body, project, agent, tags, embedding_model, created_at, updated_at,
	(SELECT slug FROM memories sb WHERE sb.id = memories.superseded_by) AS superseded_by_slug,
	(SELECT slug FROM memories ov WHERE ov.id = memories.overrides) AS overrides_slug`

type rowScanner interface {
	Scan(dest ...any) error
}

// scanMemory reads one row selected as memoryColumns; extra destinations are
// appended in order, for callers that select a trailing expression alongside
// the standard columns.
func scanMemory(row rowScanner, extra ...any) (*models.Memory, error) {
	var (
		m            models.Memory
		project      sql.NullString
		agent        sql.NullString
		supersededBy sql.NullString
		overrides    sql.NullString
	)
	dest := []any{&m.ID, &m.Type, &m.Slug, &m.Description, &m.Body, &project, &agent,
		&m.Tags, &m.EmbeddingModel, &m.CreatedAt, &m.UpdatedAt, &supersededBy, &overrides}
	err := row.Scan(append(dest, extra...)...)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, models.ErrNotFound
		}
		return nil, err
	}
	m.Project = project.String
	m.Agent = agent.String
	m.SupersededBy = supersededBy.String
	m.Overrides = overrides.String
	return &m, nil
}

// queryRower is satisfied by both *pgxpool.Pool and pgx.Tx, so idForSlug
// works inside or outside a transaction.
type queryRower interface {
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

// idForSlug resolves slug to its memory's id, or models.ErrNotFound.
func idForSlug(ctx context.Context, q queryRower, slug string) (string, error) {
	var id string
	err := q.QueryRow(ctx, `SELECT id FROM memories WHERE slug = $1`, slug).Scan(&id)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", models.ErrNotFound
	}
	return id, err
}

// nullableIDForSlug is idForSlug for an optional reference: "" resolves to
// NULL rather than ErrNotFound.
func nullableIDForSlug(ctx context.Context, q queryRower, slug string) (any, error) {
	if slug == "" {
		return nil, nil
	}
	return idForSlug(ctx, q, slug)
}

func (s *Store) CreateMemory(ctx context.Context, m NewMemory) (*models.Memory, error) {
	tags := m.Tags
	if tags == nil {
		tags = []string{}
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx)

	overridesID, err := nullableIDForSlug(ctx, tx, m.Overrides)
	if err != nil {
		return nil, err
	}
	row := tx.QueryRow(ctx, `
		INSERT INTO memories (type, slug, description, body, project, agent, tags, embedding, embedding_model, overrides)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		RETURNING `+memoryColumns,
		m.Type, m.Slug, m.Description, m.Body, nullableString(m.Project), nullableString(m.Agent),
		tags, pgvector.NewVector(m.Embedding), m.EmbeddingModel, overridesID)

	mem, err := scanMemory(row)
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" {
			return nil, models.ErrDuplicateSlug
		}
		return nil, err
	}

	if err := replaceMemoryChunks(ctx, tx, mem.ID, m.Chunks); err != nil {
		return nil, err
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return mem, nil
}

func (s *Store) UpdateMemory(ctx context.Context, slug string, p MemoryPatch) (*models.Memory, error) {
	var sets []string
	var args []any
	set := func(col string, v any) {
		args = append(args, v)
		sets = append(sets, fmt.Sprintf("%s = $%d", col, len(args)))
	}

	if p.Type != nil {
		set("type", *p.Type)
	}
	if p.Description != nil {
		set("description", *p.Description)
	}
	if p.Body != nil {
		set("body", *p.Body)
	}
	if p.Project != nil {
		set("project", nullableString(*p.Project))
	}
	if p.Agent != nil {
		set("agent", nullableString(*p.Agent))
	}
	if p.Tags != nil {
		set("tags", *p.Tags)
	}
	if p.Embedding != nil {
		set("embedding", pgvector.NewVector(p.Embedding))
	}
	if p.EmbeddingModel != nil {
		set("embedding_model", *p.EmbeddingModel)
	}
	if p.ClearSupersededBy {
		sets = append(sets, "superseded_by = NULL")
	}
	if p.Overrides != nil {
		id, err := nullableIDForSlug(ctx, s.pool, *p.Overrides)
		if err != nil {
			return nil, err
		}
		set("overrides", id)
	}
	sets = append(sets, "updated_at = now()")

	args = append(args, slug)
	query := fmt.Sprintf(`UPDATE memories SET %s WHERE slug = $%d RETURNING %s`,
		strings.Join(sets, ", "), len(args), memoryColumns)

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx)

	mem, err := scanMemory(tx.QueryRow(ctx, query, args...))
	if err != nil {
		return nil, err
	}

	if p.Chunks != nil {
		if err := replaceMemoryChunks(ctx, tx, mem.ID, *p.Chunks); err != nil {
			return nil, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return mem, nil
}

// SupersedeMemory marks oldSlug as superseded by newSlug; neither memory's
// content changes.
func (s *Store) SupersedeMemory(ctx context.Context, oldSlug, newSlug string) (*models.Memory, error) {
	newID, err := idForSlug(ctx, s.pool, newSlug)
	if err != nil {
		return nil, fmt.Errorf("new_slug %q: %w", newSlug, err)
	}
	// updated_at stays put: the memory's content didn't change, and a bump
	// would reset its decay and float it to the top of list_memories.
	row := s.pool.QueryRow(ctx, `UPDATE memories SET superseded_by = $1 WHERE slug = $2 RETURNING `+memoryColumns,
		newID, oldSlug)
	mem, err := scanMemory(row)
	if err != nil {
		return nil, fmt.Errorf("old_slug %q: %w", oldSlug, err)
	}
	return mem, nil
}

func (s *Store) GetMemoryBySlug(ctx context.Context, slug string) (*models.Memory, error) {
	row := s.pool.QueryRow(ctx, `SELECT `+memoryColumns+` FROM memories WHERE slug = $1`, slug)
	return scanMemory(row)
}

func (s *Store) DeleteMemory(ctx context.Context, slug string) error {
	tag, err := s.pool.Exec(ctx, `DELETE FROM memories WHERE slug = $1`, slug)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return models.ErrNotFound
	}
	return nil
}

func (s *Store) ListMemories(ctx context.Context, f MemoryFilter, limit, offset int) ([]models.Memory, error) {
	where, args := buildFilter(0, f)
	args = append(args, limit, offset)
	query := fmt.Sprintf(`
		SELECT %s FROM memories
		%s
		ORDER BY updated_at DESC
		LIMIT $%d OFFSET $%d
	`, memoryColumns, where, len(args)-1, len(args))

	rows, err := s.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []models.Memory
	for rows.Next() {
		m, err := scanMemory(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, *m)
	}
	return out, rows.Err()
}
