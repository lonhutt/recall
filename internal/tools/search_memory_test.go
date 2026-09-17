package tools

import (
	"context"
	"errors"
	"testing"

	"github.com/lonhutt/recall/internal/models"
	"github.com/lonhutt/recall/internal/store/postgres"
)

func TestSearchMemoryReturnsScoredResults(t *testing.T) {
	store := newFakeStore()
	store.searchResp = []postgres.ScoredMemory{
		{Memory: models.Memory{Slug: "hit-1"}, Score: 0.9},
	}
	embedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: store, Embedder: embedder}

	_, result, err := tools.SearchMemory(context.Background(), nil, SearchMemoryArgs{Query: "how do I wrap errors"})
	if err != nil {
		t.Fatalf("SearchMemory: %v", err)
	}
	if len(result.Results) != 1 || result.Results[0].Slug != "hit-1" || result.Results[0].Score != 0.9 {
		t.Errorf("Results = %+v, want one hit-1 result scored 0.9", result.Results)
	}
	if embedder.lastKind != "query" {
		t.Errorf("embedder kind = %q, want query", embedder.lastKind)
	}
}

func TestSearchMemoryEmptyResultsIsNotAnError(t *testing.T) {
	tools := &Tools{Store: newFakeStore(), Embedder: &fakeEmbedder{vector: []float32{1}}}

	_, result, err := tools.SearchMemory(context.Background(), nil, SearchMemoryArgs{Query: "nothing matches this"})
	if err != nil {
		t.Fatalf("SearchMemory: %v", err)
	}
	if len(result.Results) != 0 {
		t.Errorf("Results = %+v, want empty", result.Results)
	}
}

func TestSearchMemoryRejectsEmptyQuery(t *testing.T) {
	tools := &Tools{Store: newFakeStore(), Embedder: &fakeEmbedder{vector: []float32{1}}}

	_, _, err := tools.SearchMemory(context.Background(), nil, SearchMemoryArgs{Query: "  "})
	if err == nil {
		t.Fatal("expected an error for empty query, got nil")
	}
}

func TestSearchMemoryClampsLimit(t *testing.T) {
	cases := []struct {
		in, want int
	}{
		{0, 10},
		{-5, 1},
		{1000, 50},
		{25, 25},
	}
	for _, c := range cases {
		store := newFakeStore()
		tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}}

		if _, _, err := tools.SearchMemory(context.Background(), nil, SearchMemoryArgs{Query: "q", Limit: c.in}); err != nil {
			t.Fatalf("SearchMemory(limit=%d): %v", c.in, err)
		}
		if store.lastLimit != c.want {
			t.Errorf("SearchMemory(limit=%d) called store with limit %d, want %d", c.in, store.lastLimit, c.want)
		}
	}
}

func TestSearchMemoryWrapsEmbeddingFailure(t *testing.T) {
	embedFailure := errors.New("llama.cpp server returned 500")
	tools := &Tools{Store: newFakeStore(), Embedder: &fakeEmbedder{err: embedFailure}}

	_, _, err := tools.SearchMemory(context.Background(), nil, SearchMemoryArgs{Query: "q"})
	if !errors.Is(err, embedFailure) {
		t.Errorf("error = %v, want it to wrap %v", err, embedFailure)
	}
}
