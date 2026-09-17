package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/lonhutt/recall/internal/models"
)

type NewLogEvent struct {
	OccurredAt  time.Time
	Description string
	Project     string
	Agent       string
	Tags        []string
	Metadata    map[string]any
}

// EventFilter narrows ListLogEvents. A nil pointer means "don't filter on
// this column".
type EventFilter struct {
	Project        *string
	Agent          *string
	Tags           []string
	OccurredAfter  *time.Time
	OccurredBefore *time.Time
}

const eventColumns = "id, occurred_at, description, project, agent, tags, metadata, created_at"

func scanLogEvent(row rowScanner) (*models.LogEvent, error) {
	var (
		e        models.LogEvent
		project  sql.NullString
		agent    sql.NullString
		metadata []byte
	)
	err := row.Scan(&e.ID, &e.OccurredAt, &e.Description, &project, &agent, &e.Tags, &metadata, &e.CreatedAt)
	if err != nil {
		return nil, err
	}
	e.Project = project.String
	e.Agent = agent.String
	if len(metadata) > 0 {
		if err := json.Unmarshal(metadata, &e.Metadata); err != nil {
			return nil, fmt.Errorf("decoding event metadata: %w", err)
		}
	}
	return &e, nil
}

func (s *Store) CreateLogEvent(ctx context.Context, e NewLogEvent) (*models.LogEvent, error) {
	metadata, err := json.Marshal(e.Metadata)
	if err != nil {
		return nil, fmt.Errorf("marshaling event metadata: %w", err)
	}
	tags := e.Tags
	if tags == nil {
		tags = []string{}
	}

	row := s.pool.QueryRow(ctx, `
		INSERT INTO log_events (occurred_at, description, project, agent, tags, metadata)
		VALUES ($1, $2, $3, $4, $5, $6)
		RETURNING `+eventColumns,
		e.OccurredAt, e.Description, nullableString(e.Project), nullableString(e.Agent), tags, metadata)

	return scanLogEvent(row)
}

func (s *Store) ListLogEvents(ctx context.Context, f EventFilter, limit, offset int) ([]models.LogEvent, error) {
	var conds []string
	var args []any
	add := func(cond string, v any) {
		args = append(args, v)
		conds = append(conds, fmt.Sprintf(cond, len(args)))
	}
	if f.Project != nil {
		add("project = $%d", *f.Project)
	}
	if f.Agent != nil {
		add("agent = $%d", *f.Agent)
	}
	if len(f.Tags) > 0 {
		add("tags @> $%d", f.Tags)
	}
	if f.OccurredAfter != nil {
		add("occurred_at >= $%d", *f.OccurredAfter)
	}
	if f.OccurredBefore != nil {
		add("occurred_at <= $%d", *f.OccurredBefore)
	}

	where := ""
	if len(conds) > 0 {
		where = "WHERE " + strings.Join(conds, " AND ")
	}
	args = append(args, limit, offset)
	query := fmt.Sprintf(`
		SELECT %s FROM log_events
		%s
		ORDER BY occurred_at DESC
		LIMIT $%d OFFSET $%d
	`, eventColumns, where, len(args)-1, len(args))

	rows, err := s.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []models.LogEvent
	for rows.Next() {
		e, err := scanLogEvent(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, *e)
	}
	return out, rows.Err()
}
