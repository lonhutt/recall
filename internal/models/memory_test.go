package models_test

import (
	"testing"

	"github.com/lonhutt/recall/internal/models"
)

func TestIsValidMemoryType(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want bool
	}{
		{"user is valid", "user", true},
		{"feedback is valid", "feedback", true},
		{"project is valid", "project", true},
		{"reference is valid", "reference", true},
		{"unknown type is invalid", "bogus", false},
		{"empty string is invalid", "", false},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := models.IsValidMemoryType(c.in)
			if got != c.want {
				t.Errorf("IsValidMemoryType(%q) = %v, want %v", c.in, got, c.want)
			}
		})
	}
}
