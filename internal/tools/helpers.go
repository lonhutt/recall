package tools

import "github.com/lonhutt/recall/internal/store/postgres"

func clamp(v, min, max, def int) int {
	switch {
	case v == 0:
		return def
	case v < min:
		return min
	case v > max:
		return max
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
