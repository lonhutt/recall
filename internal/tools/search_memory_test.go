package tools

import (
	"context"
	"errors"
	"testing"
	"time"

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

func TestSearchMemoryScopesToCurrentEmbeddingModel(t *testing.T) {
	store := newFakeStore()
	embedder := &fakeEmbedder{vector: []float32{1}}
	tools := &Tools{Store: store, Embedder: embedder, EmbeddingModel: "embeddinggemma"}

	if _, _, err := tools.SearchMemory(context.Background(), nil, SearchMemoryArgs{Query: "anything"}); err != nil {
		t.Fatalf("SearchMemory: %v", err)
	}
	if store.lastFilter.EmbeddingModel == nil {
		t.Fatal("filter.EmbeddingModel = nil, want the configured model so foreign vectors are excluded")
	}
	if got := *store.lastFilter.EmbeddingModel; got != "embeddinggemma" {
		t.Errorf("filter.EmbeddingModel = %q, want embeddinggemma", got)
	}
}

func TestSearchMemoryPassesQueryTextToStore(t *testing.T) {
	store := newFakeStore()
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}}

	if _, _, err := tools.SearchMemory(context.Background(), nil, SearchMemoryArgs{Query: "RECALL_HTTP_TOKEN"}); err != nil {
		t.Fatalf("SearchMemory: %v", err)
	}
	if store.lastQuery != "RECALL_HTTP_TOKEN" {
		t.Errorf("store query = %q, want the query text passed through for keyword matching", store.lastQuery)
	}
}

func TestSearchMemoryReturnsSimilarityBesideScore(t *testing.T) {
	store := newFakeStore()
	store.searchResp = []postgres.ScoredMemory{
		{Memory: models.Memory{Slug: "hit-1"}, Score: 0.03, Similarity: 0.71},
	}
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}}

	_, result, err := tools.SearchMemory(context.Background(), nil, SearchMemoryArgs{Query: "q"})
	if err != nil {
		t.Fatalf("SearchMemory: %v", err)
	}
	if len(result.Results) != 1 || result.Results[0].Score != 0.03 || result.Results[0].Similarity != 0.71 {
		t.Errorf("Results = %+v, want score 0.03 and similarity 0.71", result.Results)
	}
}

func TestSearchMemoryReturnsOverriddenByAndAge(t *testing.T) {
	store := newFakeStore()
	store.searchResp = []postgres.ScoredMemory{
		{Memory: models.Memory{Slug: "global-style", UpdatedAt: time.Now().Add(-72 * time.Hour)}, OverriddenBy: "project-style"},
	}
	tools := &Tools{Store: store, Embedder: &fakeEmbedder{vector: []float32{1}}}

	_, result, err := tools.SearchMemory(context.Background(), nil, SearchMemoryArgs{Query: "q", Project: "recall"})
	if err != nil {
		t.Fatalf("SearchMemory: %v", err)
	}
	if len(result.Results) != 1 || result.Results[0].OverriddenBy != "project-style" || result.Results[0].Age != "3 days" {
		t.Errorf("Results = %+v, want overridden_by project-style and age 3 days", result.Results)
	}
}

func TestHumanAgeRoundsDownToWholeDays(t *testing.T) {
	now := time.Now()
	for _, c := range []struct {
		ago  time.Duration
		want string
	}{
		{time.Hour, "today"},
		{25 * time.Hour, "1 day"},
		{50 * time.Hour, "2 days"},
	} {
		if got := humanAge(now.Add(-c.ago)); got != c.want {
			t.Errorf("humanAge(%v ago) = %q, want %q", c.ago, got, c.want)
		}
	}
}
