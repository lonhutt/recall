package tools

import (
	"context"
	"fmt"

	"github.com/lonhutt/recall/internal/chunking"
	"github.com/lonhutt/recall/internal/embeddings"
	"github.com/lonhutt/recall/internal/store/postgres"
)

// embedDescription embeds description alone as a memory's primary search
// vector. A description is documented as a one-line summary, so unlike
// body it's never expected to need chunking.
func (t *Tools) embedDescription(ctx context.Context, description string) ([]float32, error) {
	vectors, err := t.Embedder.Embed(ctx, []string{description}, embeddings.InputTypeDocument)
	if err != nil {
		return nil, err
	}
	return vectors[0], nil
}

// chunkAndEmbedBody splits body into pieces that fit t.MaxTokensPerChunk
// and embeds each one, for the secondary chunk-level search layer. A body
// that already fits produces exactly one chunk, and t.MaxTokensPerChunk <= 0
// disables chunking altogether (the same "no limit" convention llamacpp.New
// documents for the config value both share).
//
// The chunks themselves stay unprefixed — Embed applies its own prefix — but
// the token budget is measured against t.DocumentPrefix+text, because that's
// the string Embed will check against the model's context window.
func (t *Tools) chunkAndEmbedBody(ctx context.Context, body string) ([]postgres.MemoryChunk, error) {
	var texts []string
	if t.MaxTokensPerChunk <= 0 {
		texts = []string{body}
	} else {
		counter := func(ctx context.Context, text string) (int, error) {
			return t.Embedder.CountTokens(ctx, t.DocumentPrefix+text)
		}
		var err error
		texts, err = chunking.Chunk(ctx, body, t.MaxTokensPerChunk, counter)
		if err != nil {
			return nil, fmt.Errorf("chunking body: %w", err)
		}
	}
	vectors, err := t.Embedder.Embed(ctx, texts, embeddings.InputTypeDocument)
	if err != nil {
		return nil, err
	}
	chunks := make([]postgres.MemoryChunk, len(texts))
	for i, text := range texts {
		chunks[i] = postgres.MemoryChunk{ChunkIndex: i, Text: text, Embedding: vectors[i]}
	}
	return chunks, nil
}

// BackfillMemoryEmbedding regenerates slug's description embedding and
// body chunks from scratch. It's for the backfill-chunks command, to
// migrate memories saved before chunked search existed; normal save/update
// flows never need to call it directly.
func (t *Tools) BackfillMemoryEmbedding(ctx context.Context, slug, description, body string) error {
	descVec, err := t.embedDescription(ctx, description)
	if err != nil {
		return fmt.Errorf("generating description embedding: %w", err)
	}
	chunks, err := t.chunkAndEmbedBody(ctx, body)
	if err != nil {
		return fmt.Errorf("generating body chunk embeddings: %w", err)
	}
	_, err = t.Store.UpdateMemory(ctx, slug, postgres.MemoryPatch{
		Embedding:      descVec,
		EmbeddingModel: &t.EmbeddingModel,
		Chunks:         &chunks,
	})
	return err
}
