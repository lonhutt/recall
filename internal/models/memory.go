package models

import "time"

type MemoryType string

const (
	MemoryTypeUser      MemoryType = "user"
	MemoryTypeFeedback  MemoryType = "feedback"
	MemoryTypeProject   MemoryType = "project"
	MemoryTypeReference MemoryType = "reference"
)

func ValidMemoryTypes() []MemoryType {
	return []MemoryType{MemoryTypeUser, MemoryTypeFeedback, MemoryTypeProject, MemoryTypeReference}
}

func IsValidMemoryType(t string) bool {
	for _, v := range ValidMemoryTypes() {
		if string(v) == t {
			return true
		}
	}
	return false
}

type Memory struct {
	ID             string
	Type           string
	Slug           string
	Description    string
	Body           string
	Project        string
	Agent          string
	Tags           []string
	EmbeddingModel string
	CreatedAt      time.Time
	UpdatedAt      time.Time
}
