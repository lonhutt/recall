package models

import "time"

// EmbeddingDimension is the fixed vector width used for every stored
// embedding, matching EmbeddingGemma's native output. Changing it requires
// a schema migration and re-embedding all existing rows.
const EmbeddingDimension = 768

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
