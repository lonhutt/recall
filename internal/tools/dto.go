package tools

import (
	"time"

	"github.com/lonhutt/recall/internal/models"
)

type MemoryDTO struct {
	ID          string   `json:"id"`
	Type        string   `json:"type"`
	Slug        string   `json:"slug"`
	Description string   `json:"description"`
	Body        string   `json:"body"`
	Project     string   `json:"project,omitempty"`
	Agent       string   `json:"agent,omitempty"`
	Tags        []string `json:"tags,omitempty"`
	CreatedAt   string   `json:"created_at"`
	UpdatedAt   string   `json:"updated_at"`
}

func newMemoryDTO(m *models.Memory) MemoryDTO {
	return MemoryDTO{
		ID:          m.ID,
		Type:        m.Type,
		Slug:        m.Slug,
		Description: m.Description,
		Body:        m.Body,
		Project:     m.Project,
		Agent:       m.Agent,
		Tags:        m.Tags,
		CreatedAt:   m.CreatedAt.Format(time.RFC3339),
		UpdatedAt:   m.UpdatedAt.Format(time.RFC3339),
	}
}

type ScoredMemoryDTO struct {
	MemoryDTO
	Score float64 `json:"score"`
}

type LogEventDTO struct {
	ID          string         `json:"id"`
	OccurredAt  string         `json:"occurred_at"`
	Description string         `json:"description"`
	Project     string         `json:"project,omitempty"`
	Agent       string         `json:"agent,omitempty"`
	Tags        []string       `json:"tags,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

func newLogEventDTO(e *models.LogEvent) LogEventDTO {
	return LogEventDTO{
		ID:          e.ID,
		OccurredAt:  e.OccurredAt.Format(time.RFC3339),
		Description: e.Description,
		Project:     e.Project,
		Agent:       e.Agent,
		Tags:        e.Tags,
		Metadata:    e.Metadata,
	}
}
