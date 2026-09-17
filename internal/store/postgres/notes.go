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
}

const memoryColumns = "id, type, slug, description, body, project, agent, tags, embedding_model, created_at, updated_at"

type rowScanner interface {
	Scan(dest ...any) error
}

// scanMemory reads one row selected as memoryColumns; extra destinations are
// appended in order, for callers that select a trailing expression alongside
// the standard columns.
func scanMemory(row rowScanner, extra ...any) (*models.Memory, error) {
	var (
		m       models.Memory
		project sql.NullString
		agent   sql.NullString
	)
	dest := []any{&m.ID, &m.Type, &m.Slug, &m.Description, &m.Body, &project, &agent,
		&m.Tags, &m.EmbeddingModel, &m.CreatedAt, &m.UpdatedAt}
	err := row.Scan(append(dest, extra...)...)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, models.ErrNotFound
		}
		return nil, err
	}
	m.Project = project.String
	m.Agent = agent.String
	return &m, nil
}

func (s *Store) CreateMemory(ctx context.Context, m NewMemory) (*models.Memory, error) {
	tags := m.Tags
	if tags == nil {
		tags = []string{}
	}
	row := s.pool.QueryRow(ctx, `
		INSERT INTO memories (type, slug, description, body, project, agent, tags, embedding, embedding_model)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		RETURNING `+memoryColumns,
		m.Type, m.Slug, m.Description, m.Body, nullableString(m.Project), nullableString(m.Agent),
		tags, pgvector.NewVector(m.Embedding), m.EmbeddingModel)

	mem, err := scanMemory(row)
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" {
			return nil, models.ErrDuplicateSlug
		}
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
	sets = append(sets, "updated_at = now()")

	args = append(args, slug)
	query := fmt.Sprintf(`UPDATE memories SET %s WHERE slug = $%d RETURNING %s`,
		strings.Join(sets, ", "), len(args), memoryColumns)

	return scanMemory(s.pool.QueryRow(ctx, query, args...))
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
