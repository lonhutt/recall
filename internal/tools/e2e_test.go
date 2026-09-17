package tools

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// TestMCPServerExposesAllToolsEndToEnd drives all 7 tools over a real MCP
// JSON-RPC session (in-memory transport), through a real mcp.Client and
// mcp.Server, exercising the actual schema inference/validation path that
// RegisterAll wires up. The store/embedder are still fakes: this test is
// about the MCP wiring, not the SQL layer (covered separately by the
// postgres integration tests).
func TestMCPServerExposesAllToolsEndToEnd(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1, 0, 0}}
	toolset := &Tools{Store: store, Embedder: embedder, EmbeddingModel: "embeddinggemma"}

	server := mcp.NewServer(&mcp.Implementation{Name: "recall-test", Version: "test"}, nil)
	toolset.RegisterAll(server)

	ctx := context.Background()
	serverTransport, clientTransport := mcp.NewInMemoryTransports()

	serverSession, err := server.Connect(ctx, serverTransport, nil)
	if err != nil {
		t.Fatalf("server.Connect: %v", err)
	}
	defer serverSession.Close()

	client := mcp.NewClient(&mcp.Implementation{Name: "recall-e2e-client", Version: "test"}, nil)
	session, err := client.Connect(ctx, clientTransport, nil)
	if err != nil {
		t.Fatalf("client.Connect: %v", err)
	}
	defer session.Close()

	call := func(name string, args any) map[string]any {
		t.Helper()
		res, err := session.CallTool(ctx, &mcp.CallToolParams{Name: name, Arguments: args})
		if err != nil {
			t.Fatalf("CallTool(%s): %v", name, err)
		}
		if res.IsError {
			t.Fatalf("CallTool(%s) returned a tool error: %v", name, res.GetError())
		}
		b, err := json.Marshal(res.StructuredContent)
		if err != nil {
			t.Fatalf("marshaling structured content for %s: %v", name, err)
		}
		var out map[string]any
		if err := json.Unmarshal(b, &out); err != nil {
			t.Fatalf("unmarshaling structured content for %s: %v", name, err)
		}
		return out
	}

	saved := call("save_memory", SaveMemoryArgs{
		Type: "reference", Slug: "e2e-note", Description: "old desc", Body: "old body",
	})
	if mem := saved["memory"].(map[string]any); mem["slug"] != "e2e-note" {
		t.Fatalf("save_memory result: %+v", saved)
	}

	searched := call("search_memory", SearchMemoryArgs{Query: "anything"})
	results := searched["results"].([]any)
	if len(results) != 1 || results[0].(map[string]any)["slug"] != "e2e-note" {
		t.Fatalf("search_memory result: %+v", searched)
	}

	listed := call("list_memories", ListMemoriesArgs{})
	memories := listed["memories"].([]any)
	if len(memories) != 1 || memories[0].(map[string]any)["slug"] != "e2e-note" {
		t.Fatalf("list_memories result: %+v", listed)
	}

	newDesc := "new desc"
	updated := call("update_memory", UpdateMemoryArgs{Slug: "e2e-note", Description: &newDesc})
	if mem := updated["memory"].(map[string]any); mem["description"] != "new desc" {
		t.Fatalf("update_memory result: %+v", updated)
	}

	searchedAgain := call("search_memory", SearchMemoryArgs{Query: "anything"})
	resultsAgain := searchedAgain["results"].([]any)
	if len(resultsAgain) != 1 || resultsAgain[0].(map[string]any)["description"] != "new desc" {
		t.Fatalf("second search_memory result: %+v", searchedAgain)
	}

	deleted := call("delete_memory", DeleteMemoryArgs{Slug: "e2e-note"})
	if deleted["deleted"] != true {
		t.Fatalf("delete_memory result: %+v", deleted)
	}

	loggedEvent := call("log_event", LogEventArgs{Description: "e2e ran"})
	if ev := loggedEvent["event"].(map[string]any); ev["description"] != "e2e ran" {
		t.Fatalf("log_event result: %+v", loggedEvent)
	}

	events := call("list_events", ListEventsArgs{})
	eventList := events["events"].([]any)
	if len(eventList) != 1 || eventList[0].(map[string]any)["description"] != "e2e ran" {
		t.Fatalf("list_events result: %+v", events)
	}
}
