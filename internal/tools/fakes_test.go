package tools

import (
	"context"
	"strings"
	"testing"

	"github.com/lonhutt/recall/internal/embeddings"
	"github.com/lonhutt/recall/internal/models"
	"github.com/lonhutt/recall/internal/store/postgres"
)

// fakeStore is an in-memory Store double keyed by slug, for tool-handler
// unit tests that shouldn't need a real database.
type fakeStore struct {
	bySlug     map[string]models.Memory
	events     []models.LogEvent
	createErr  error
	updateErr  error
	getErr     error
	deleteErr  error
	searchResp []postgres.ScoredMemory
	// similarResp is what FindSimilarMemory returns; nil means no duplicate.
	similarResp *postgres.ScoredMemory
	searchErr   error
	lastLimit   int
	lastFilter  postgres.MemoryFilter
	lastQuery   string
	lastChunks  []postgres.MemoryChunk
}

func newFakeStore() *fakeStore {
	return &fakeStore{bySlug: map[string]models.Memory{}}
}

func (f *fakeStore) CreateMemory(_ context.Context, m postgres.NewMemory) (*models.Memory, error) {
	if f.createErr != nil {
		return nil, f.createErr
	}
	f.lastChunks = m.Chunks
	if _, exists := f.bySlug[m.Slug]; exists {
		return nil, models.ErrDuplicateSlug
	}
	mem := models.Memory{
		ID: m.Slug, Type: m.Type, Slug: m.Slug, Description: m.Description, Body: m.Body,
		Project: m.Project, Agent: m.Agent, Tags: m.Tags, EmbeddingModel: m.EmbeddingModel,
		Overrides: m.Overrides,
	}
	f.bySlug[m.Slug] = mem
	return &mem, nil
}

func (f *fakeStore) UpdateMemory(_ context.Context, slug string, p postgres.MemoryPatch) (*models.Memory, error) {
	if f.updateErr != nil {
		return nil, f.updateErr
	}
	if p.Chunks != nil {
		f.lastChunks = *p.Chunks
	}
	mem, ok := f.bySlug[slug]
	if !ok {
		return nil, models.ErrNotFound
	}
	if p.Type != nil {
		mem.Type = *p.Type
	}
	if p.Description != nil {
		mem.Description = *p.Description
	}
	if p.Body != nil {
		mem.Body = *p.Body
	}
	if p.Project != nil {
		mem.Project = *p.Project
	}
	if p.Agent != nil {
		mem.Agent = *p.Agent
	}
	if p.Tags != nil {
		mem.Tags = *p.Tags
	}
	if p.EmbeddingModel != nil {
		mem.EmbeddingModel = *p.EmbeddingModel
	}
	if p.ClearSupersededBy {
		mem.SupersededBy = ""
	}
	if p.Overrides != nil {
		mem.Overrides = *p.Overrides
	}
	f.bySlug[slug] = mem
	return &mem, nil
}

func (f *fakeStore) GetMemoryBySlug(_ context.Context, slug string) (*models.Memory, error) {
	if f.getErr != nil {
		return nil, f.getErr
	}
	mem, ok := f.bySlug[slug]
	if !ok {
		return nil, models.ErrNotFound
	}
	return &mem, nil
}

func (f *fakeStore) DeleteMemory(_ context.Context, slug string) error {
	if f.deleteErr != nil {
		return f.deleteErr
	}
	if _, ok := f.bySlug[slug]; !ok {
		return models.ErrNotFound
	}
	delete(f.bySlug, slug)
	return nil
}

func (f *fakeStore) ListMemories(_ context.Context, _ postgres.MemoryFilter, limit, _ int) ([]models.Memory, error) {
	f.lastLimit = limit
	var out []models.Memory
	for _, m := range f.bySlug {
		out = append(out, m)
	}
	return out, nil
}

// SearchMemories returns f.searchResp verbatim when a test has set it
// explicitly (for asserting exact scores/ordering); otherwise it derives
// results from bySlug applying filter, so end-to-end flows that save then
// search see their own data.
func (f *fakeStore) SearchMemories(_ context.Context, _ []float32, query string, filter postgres.MemoryFilter, limit int) ([]postgres.ScoredMemory, error) {
	f.lastQuery = query
	f.lastLimit = limit
	f.lastFilter = filter
	if f.searchErr != nil {
		return nil, f.searchErr
	}
	if f.searchResp != nil {
		return f.searchResp, nil
	}
	var out []postgres.ScoredMemory
	for _, m := range f.bySlug {
		if filter.Type != nil && m.Type != *filter.Type {
			continue
		}
		// Like the real store, a project search also sees global memories.
		if filter.Project != nil && m.Project != *filter.Project && m.Project != "" {
			continue
		}
		if filter.Agent != nil && m.Agent != *filter.Agent {
			continue
		}
		if filter.EmbeddingModel != nil && m.EmbeddingModel != *filter.EmbeddingModel {
			continue
		}
		out = append(out, postgres.ScoredMemory{Memory: m, Score: 1})
	}
	return out, nil
}

func (f *fakeStore) CreateLogEvent(_ context.Context, e postgres.NewLogEvent) (*models.LogEvent, error) {
	event := models.LogEvent{
		OccurredAt: e.OccurredAt, Description: e.Description, Project: e.Project,
		Agent: e.Agent, Tags: e.Tags, Metadata: e.Metadata,
	}
	f.events = append(f.events, event)
	return &event, nil
}

func (f *fakeStore) ListLogEvents(_ context.Context, _ postgres.EventFilter, _, _ int) ([]models.LogEvent, error) {
	return f.events, nil
}

// embedCall records one Embed invocation, for tests that need to assert on
// more than just the most recent call (e.g. a tool call that embeds a
// description and then, separately, one or more body chunks).
type embedCall struct {
	texts []string
	kind  embeddings.InputType
}

// fakeEmbedder is an Embedder double returning a fixed vector (or error) per call.
type fakeEmbedder struct {
	vector        []float32
	err           error
	lastKind      embeddings.InputType
	lastTexts     []string
	calls         int
	allCalls      []embedCall
	countTokensFn func(text string) (int, error)
}

func (f *fakeEmbedder) Embed(_ context.Context, texts []string, kind embeddings.InputType) ([][]float32, error) {
	f.calls++
	f.lastTexts = texts
	f.lastKind = kind
	f.allCalls = append(f.allCalls, embedCall{texts: texts, kind: kind})
	if f.err != nil {
		return nil, f.err
	}
	vectors := make([][]float32, len(texts))
	for i := range texts {
		vectors[i] = f.vector
	}
	return vectors, nil
}

// CountTokens is a fake tokenizer for tests: one token per word, so tests
// needing deterministic chunk boundaries can reason about expected output
// without a real tokenizer. countTokensFn overrides this when a test needs
// specific counts.
func (f *fakeEmbedder) CountTokens(_ context.Context, text string) (int, error) {
	if f.countTokensFn != nil {
		return f.countTokensFn(text)
	}
	return len(strings.Fields(text)), nil
}

func TestFakeEmbedderSatisfiesWidenedEmbedderInterface(t *testing.T) {
	var _ Embedder = (*fakeEmbedder)(nil)
}

func (f *fakeStore) SupersedeMemory(_ context.Context, oldSlug, newSlug string) (*models.Memory, error) {
	mem, ok := f.bySlug[oldSlug]
	if !ok {
		return nil, models.ErrNotFound
	}
	if _, ok := f.bySlug[newSlug]; !ok {
		return nil, models.ErrNotFound
	}
	mem.SupersededBy = newSlug
	f.bySlug[oldSlug] = mem
	return &mem, nil
}

func (f *fakeStore) FindSimilarMemory(_ context.Context, _ []float32, _, _, _ string, _ float64) (*postgres.ScoredMemory, error) {
	return f.similarResp, nil
}
