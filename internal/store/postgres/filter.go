package postgres

import (
	"fmt"
	"strings"
)

// MemoryFilter narrows both ListMemories and SearchMemories by metadata.
// A nil pointer field means "don't filter on this column".
type MemoryFilter struct {
	Type    *string
	Project *string
	Agent   *string
	Tags    []string
}

// buildFilter renders f as a SQL WHERE clause (empty string if f has no
// conditions) whose placeholders start at paramOffset+1, plus the matching
// args in order.
func buildFilter(paramOffset int, f MemoryFilter) (where string, args []any) {
	var conds []string
	i := paramOffset

	add := func(col string, v *string) {
		if v == nil {
			return
		}
		i++
		conds = append(conds, fmt.Sprintf("%s = $%d", col, i))
		args = append(args, *v)
	}
	add("type", f.Type)
	add("project", f.Project)
	add("agent", f.Agent)

	if len(f.Tags) > 0 {
		i++
		conds = append(conds, fmt.Sprintf("tags @> $%d", i))
		args = append(args, f.Tags)
	}

	if len(conds) == 0 {
		return "", args
	}
	return "WHERE " + strings.Join(conds, " AND "), args
}

// nullableString maps an empty string to SQL NULL, matching how memories.project
// and memories.agent distinguish "unset" from "".
func nullableString(s string) any {
	if s == "" {
		return nil
	}
	return s
}
