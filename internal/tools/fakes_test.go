package tools

import (
	"context"

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
	searchErr  error
	lastLimit  int
}

func newFakeStore() *fakeStore {
	return &fakeStore{bySlug: map[string]models.Memory{}}
}

func (f *fakeStore) CreateMemory(_ context.Context, m postgres.NewMemory) (*models.Memory, error) {
	if f.createErr != nil {
		return nil, f.createErr
	}
	if _, exists := f.bySlug[m.Slug]; exists {
		return nil, models.ErrDuplicateSlug
	}
	mem := models.Memory{
		ID: m.Slug, Type: m.Type, Slug: m.Slug, Description: m.Description, Body: m.Body,
		Project: m.Project, Agent: m.Agent, Tags: m.Tags, EmbeddingModel: m.EmbeddingModel,
	}
	f.bySlug[m.Slug] = mem
	return &mem, nil
}

func (f *fakeStore) UpdateMemory(_ context.Context, slug string, p postgres.MemoryPatch) (*models.Memory, error) {
	if f.updateErr != nil {
		return nil, f.updateErr
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
func (f *fakeStore) SearchMemories(_ context.Context, _ []float32, filter postgres.MemoryFilter, limit int) ([]postgres.ScoredMemory, error) {
	f.lastLimit = limit
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
		if filter.Project != nil && m.Project != *filter.Project {
			continue
		}
		if filter.Agent != nil && m.Agent != *filter.Agent {
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

// fakeEmbedder is an Embedder double returning a fixed vector (or error) per call.
type fakeEmbedder struct {
	vector    []float32
	err       error
	lastKind  embeddings.InputType
	lastTexts []string
	calls     int
}

func (f *fakeEmbedder) Embed(_ context.Context, texts []string, kind embeddings.InputType) ([][]float32, error) {
	f.calls++
	f.lastTexts = texts
	f.lastKind = kind
	if f.err != nil {
		return nil, f.err
	}
	vectors := make([][]float32, len(texts))
	for i := range texts {
		vectors[i] = f.vector
	}
	return vectors, nil
}
