package tools

import (
	"context"
	"strings"
	"testing"
)

// chunkAndEmbedBody has to measure its token budget against the string
// Embed will actually check — the chunk with the embedder's document prefix
// prepended. Measuring the raw chunk instead lets pack() fill a chunk right
// up to the budget, only for Embed to reject it once the prefix pushes it
// over (with a "split it into multiple memories" message that makes no
// sense, since chunking already ran).
func TestChunkAndEmbedBodyCountsTokensWithTheDocumentPrefix(t *testing.T) {
	const prefix = "prefix: "
	const maxTokens = 5

	var counted []string
	embedder := &fakeEmbedder{
		vector: []float32{1},
		countTokensFn: func(text string) (int, error) {
			counted = append(counted, text)
			return len(strings.Fields(text)), nil
		},
	}
	tools := &Tools{Store: newFakeStore(), Embedder: embedder, MaxTokensPerChunk: maxTokens, DocumentPrefix: prefix}

	body := "one two three four five six seven eight nine ten eleven twelve"
	chunks, err := tools.chunkAndEmbedBody(context.Background(), body)
	if err != nil {
		t.Fatalf("chunkAndEmbedBody: %v", err)
	}
	if len(counted) == 0 {
		t.Fatal("token counter was never called")
	}
	for _, text := range counted {
		if !strings.HasPrefix(text, prefix) {
			t.Errorf("counted %q, want the document prefix included; otherwise the budget check measures a different string than Embed does", text)
		}
	}

	for i, c := range chunks {
		if strings.HasPrefix(c.Text, prefix) {
			t.Errorf("chunk[%d].Text = %q, want the raw body text — Embed prepends the prefix itself, so storing it would double it", i, c.Text)
		}
		if n := len(strings.Fields(prefix + c.Text)); n > maxTokens {
			t.Errorf("chunk[%d] is %d tokens once prefixed, want <= %d; Embed would reject it", i, n, maxTokens)
		}
	}
}

// RECALL_EMBEDDING_MAX_TOKENS <= 0 is documented (on llamacpp.New) as
// disabling the token-limit check. Chunking shares that config value, so it
// has to read it the same way — chunking.Chunk rejects a non-positive
// budget outright, which would turn the documented escape hatch into a
// failure on every save.
func TestChunkAndEmbedBodySkipsChunkingWhenTheBudgetIsDisabled(t *testing.T) {
	embedder := &fakeEmbedder{
		vector: []float32{1},
		countTokensFn: func(string) (int, error) {
			t.Error("CountTokens called although the token budget is disabled")
			return 0, nil
		},
	}
	tools := &Tools{Store: newFakeStore(), Embedder: embedder, MaxTokensPerChunk: 0}

	body := strings.TrimSpace(strings.Repeat("word ", 5000))
	chunks, err := tools.chunkAndEmbedBody(context.Background(), body)
	if err != nil {
		t.Fatalf("chunkAndEmbedBody: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want exactly 1 (unchunked) when the budget is disabled", len(chunks))
	}
	if chunks[0].Text != body {
		t.Errorf("chunk[0].Text is %d bytes, want the full %d-byte body unchanged", len(chunks[0].Text), len(body))
	}
	if chunks[0].ChunkIndex != 0 {
		t.Errorf("chunk[0].ChunkIndex = %d, want 0", chunks[0].ChunkIndex)
	}
	if len(chunks[0].Embedding) == 0 {
		t.Error("chunk[0] has no embedding")
	}
}
