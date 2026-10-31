package postgres

import (
	"math"
	"testing"
	"time"
)

func TestDecayFactorHalvesAtEachTypesHalfLife(t *testing.T) {
	now := time.Now()
	cases := []struct {
		memType string
		age     time.Duration
	}{
		{"user", 720 * 24 * time.Hour},
		{"reference", 720 * 24 * time.Hour},
		{"project", 90 * 24 * time.Hour},
		{"feedback", 90 * 24 * time.Hour},
	}
	for _, c := range cases {
		if got := decayFactor(c.memType, now.Add(-c.age)); math.Abs(got-0.5) > 1e-3 {
			t.Errorf("decayFactor(%q, %v ago) = %v, want ~0.5", c.memType, c.age, got)
		}
	}
}
