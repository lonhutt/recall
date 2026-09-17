package tools

import (
	"context"
	"testing"
)

func TestListMemoriesReturnsStoredMemories(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: store, Embedder: embedder}
	seedMemory(t, store, embedder, tools)

	_, result, err := tools.ListMemories(context.Background(), nil, ListMemoriesArgs{})
	if err != nil {
		t.Fatalf("ListMemories: %v", err)
	}
	if len(result.Memories) != 1 || result.Memories[0].Slug != "seeded" {
		t.Errorf("Memories = %+v, want one seeded memory", result.Memories)
	}
}

func TestListMemoriesDoesNotCallEmbedder(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: store, Embedder: embedder}
	seedMemory(t, store, embedder, tools)
	embedder.calls = 0

	if _, _, err := tools.ListMemories(context.Background(), nil, ListMemoriesArgs{}); err != nil {
		t.Fatalf("ListMemories: %v", err)
	}
	if embedder.calls != 0 {
		t.Errorf("embedder called %d times, want 0 (list_memories is not semantic search)", embedder.calls)
	}
}

func TestListMemoriesClampsLimit(t *testing.T) {
	cases := []struct {
		in, want int
	}{
		{0, 50},
		{-5, 1},
		{1000, 200},
		{75, 75},
	}
	for _, c := range cases {
		store := newFakeStore()
		tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}}

		if _, _, err := tools.ListMemories(context.Background(), nil, ListMemoriesArgs{Limit: c.in}); err != nil {
			t.Fatalf("ListMemories(limit=%d): %v", c.in, err)
		}
		if store.lastLimit != c.want {
			t.Errorf("ListMemories(limit=%d) called store with limit %d, want %d", c.in, store.lastLimit, c.want)
		}
	}
}
