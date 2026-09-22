package chunking

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// wordCounter is a fake TokenCounter for tests: one token per word, so
// expected outputs are easy to reason about without a real tokenizer.
func wordCounter(_ context.Context, text string) (int, error) {
	return len(strings.Fields(text)), nil
}

func TestChunkReturnsTextUnchangedWhenWithinBudget(t *testing.T) {
	got, err := Chunk(context.Background(), "one two three", 10, wordCounter)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(got) != 1 || got[0] != "one two three" {
		t.Errorf("Chunk() = %v, want a single unchanged chunk", got)
	}
}

func TestChunkSplitsOnParagraphBoundariesWhenOverBudget(t *testing.T) {
	text := "one two three\n\nfour five six\n\nseven eight nine"
	got, err := Chunk(context.Background(), text, 6, wordCounter)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("Chunk() returned %d chunks, want 2 (paragraphs packed two-then-one): %v", len(got), got)
	}
	if got[0] != "one two three\n\nfour five six" {
		t.Errorf("chunk[0] = %q, want the first two paragraphs packed together", got[0])
	}
	if got[1] != "seven eight nine" {
		t.Errorf("chunk[1] = %q, want the third paragraph alone", got[1])
	}
}

func TestChunkEveryPieceFitsWithinMaxTokens(t *testing.T) {
	text := strings.Repeat("word ", 500)
	const maxTokens = 20
	got, err := Chunk(context.Background(), text, maxTokens, wordCounter)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	for i, c := range got {
		n, _ := wordCounter(context.Background(), c)
		if n > maxTokens {
			t.Errorf("chunk[%d] has %d tokens, want <= %d", i, n, maxTokens)
		}
	}
}

func TestChunkFallsBackToSentencesForAnOversizedParagraph(t *testing.T) {
	text := "one two. three four. five six. seven eight."
	got, err := Chunk(context.Background(), text, 4, wordCounter)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	for i, c := range got {
		if n, _ := wordCounter(context.Background(), c); n > 4 {
			t.Errorf("chunk[%d] = %q has %d tokens, want <= 4", i, c, n)
		}
	}
	rejoined := strings.Join(got, " ")
	for _, want := range []string{"one two.", "three four.", "five six.", "seven eight."} {
		if !strings.Contains(rejoined, want) {
			t.Errorf("rejoined chunks %q missing sentence %q", rejoined, want)
		}
	}
}

func TestChunkFallsBackToWordsForAnOversizedSentence(t *testing.T) {
	longRun := strings.TrimSpace(strings.Repeat("word ", 20))
	text := "intro. " + longRun
	const maxTokens = 5
	got, err := Chunk(context.Background(), text, maxTokens, wordCounter)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(got) < 2 {
		t.Fatalf("Chunk() = %v, want the long word run split into multiple chunks", got)
	}
	for i, c := range got {
		if n, _ := wordCounter(context.Background(), c); n > maxTokens {
			t.Errorf("chunk[%d] = %q has more than %d tokens", i, c, maxTokens)
		}
	}
}

func TestChunkTerminatesOnAnIndivisibleOversizedWord(t *testing.T) {
	text := strings.Repeat("x", 50)
	got, err := Chunk(context.Background(), text, 1, wordCounter)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(got) != 1 || got[0] != text {
		t.Errorf("Chunk() = %v, want the single indivisible run accepted as one oversized chunk", got)
	}
}

// A paragraph or sentence with no '.'/'!'/'?' — a bullet-style line, a
// run-on clause, a URL-heavy line — must still fall through to word-level
// splitting rather than being treated as indivisible just because the
// sentence-fallback tier found no punctuation to split on.
func TestChunkFallsBackToWordsWhenParagraphHasNoSentencePunctuation(t *testing.T) {
	text := strings.TrimSpace(strings.Repeat("word ", 20))
	const maxTokens = 5
	got, err := Chunk(context.Background(), text, maxTokens, wordCounter)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(got) < 2 {
		t.Fatalf("Chunk() = %v, want the punctuation-free run split by word", got)
	}
	for i, c := range got {
		if n, _ := wordCounter(context.Background(), c); n > maxTokens {
			t.Errorf("chunk[%d] = %q has %d tokens, want <= %d", i, c, n, maxTokens)
		}
	}
}

func TestChunkPropagatesCounterError(t *testing.T) {
	boom := errors.New("boom")
	failing := func(context.Context, string) (int, error) { return 0, boom }
	if _, err := Chunk(context.Background(), "hello", 10, failing); !errors.Is(err, boom) {
		t.Errorf("Chunk() error = %v, want it to wrap %v", err, boom)
	}
}
