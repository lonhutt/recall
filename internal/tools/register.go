package tools

import "github.com/modelcontextprotocol/go-sdk/mcp"

// RegisterAll wires every Recall tool onto s.
func (t *Tools) RegisterAll(s *mcp.Server) {
	mcp.AddTool(s, &mcp.Tool{
		Name:        "save_memory",
		Description: "Save a new structured memory note (a curated, typed, slugged piece of knowledge).",
	}, t.SaveMemory)
	mcp.AddTool(s, &mcp.Tool{
		Name:        "update_memory",
		Description: "Update fields of an existing memory note by slug. Only given fields change.",
	}, t.UpdateMemory)
	mcp.AddTool(s, &mcp.Tool{
		Name:        "search_memory",
		Description: "Semantically search memory notes by natural-language query, with optional metadata filters.",
	}, t.SearchMemory)
	mcp.AddTool(s, &mcp.Tool{
		Name:        "list_memories",
		Description: "List/browse memory notes by metadata filters, without semantic search.",
	}, t.ListMemories)
	mcp.AddTool(s, &mcp.Tool{
		Name:        "delete_memory",
		Description: "Delete a memory note by slug.",
	}, t.DeleteMemory)
	mcp.AddTool(s, &mcp.Tool{
		Name:        "log_event",
		Description: "Record a timestamped episodic event.",
	}, t.LogEvent)
	mcp.AddTool(s, &mcp.Tool{
		Name:        "list_events",
		Description: "List episodic events by metadata and date-range filters.",
	}, t.ListEvents)
}
