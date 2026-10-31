package tools

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/lonhutt/recall/internal/models"
	"github.com/lonhutt/recall/internal/store/postgres"
)

// limitOrDefault resolves a caller-supplied result limit; absent (0) becomes
// def, and anything out of range gets pulled to the nearest bound.
func limitOrDefault(v, low, high, def int) int {
	switch {
	case v == 0:
		return def
	case v < low:
		return low
	case v > high:
		return high
	default:
		return v
	}
}

func memoryFilterFrom(typ, project, agent string, tags []string) postgres.MemoryFilter {
	f := postgres.MemoryFilter{Tags: tags}
	if typ != "" {
		f.Type = &typ
	}
	if project != "" {
		f.Project = &project
	}
	if agent != "" {
		f.Agent = &agent
	}
	return f
}

// validateOverrides checks overrides is a legal project-overrides-global
// relationship: the declaring memory is project-scoped and the target exists
// and is global. Only that shape is allowed, which also rules out cycles.
func (t *Tools) validateOverrides(ctx context.Context, project, overrides string) error {
	if overrides == "" {
		return nil
	}
	if project == "" {
		return fmt.Errorf("overrides requires this memory to be project-scoped; global memories can't override anything")
	}
	target, err := t.Store.GetMemoryBySlug(ctx, overrides)
	if err != nil {
		if errors.Is(err, models.ErrNotFound) {
			return fmt.Errorf("overrides target %q not found", overrides)
		}
		return err
	}
	if target.Project != "" {
		return fmt.Errorf("overrides target %q is project-scoped (%q); overrides only applies to global memories", overrides, target.Project)
	}
	return nil
}

// humanAge renders how long ago t was in whole days, so an agent sees
// staleness without doing timestamp arithmetic.
func humanAge(t time.Time) string {
	days := int(time.Since(t).Hours() / 24)
	switch {
	case days <= 0:
		return "today"
	case days == 1:
		return "1 day"
	default:
		return fmt.Sprintf("%d days", days)
	}
}
