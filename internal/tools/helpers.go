package tools

import "github.com/lonhutt/recall/internal/store/postgres"

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
