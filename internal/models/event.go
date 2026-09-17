package models

import "time"

type LogEvent struct {
	ID          string
	OccurredAt  time.Time
	Description string
	Project     string
	Agent       string
	Tags        []string
	Metadata    map[string]any
	CreatedAt   time.Time
}
