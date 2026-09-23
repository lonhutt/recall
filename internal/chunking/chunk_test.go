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

// charCounter is a fake TokenCounter that counts every character as one
// token — unlike wordCounter, it can report more than 1 token for a single
// space-free run, which is what's needed to actually exercise pack()'s
// "genuinely indivisible" branch (a whitespace-based counter can never see
// a space-free run as oversized, since it always counts as 1 token).
func charCounter(_ context.Context, text string) (int, error) {
	return len(text), nil
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

// A long space-free run is one token to a whitespace-based counter, so this
// exits at Chunk's "already fits" check and never reaches pack(). The test
// below it is the one that exercises pack()'s indivisible branch.
func TestChunkReturnsTextUnchangedWhenTheCounterSeesItAsOneToken(t *testing.T) {
	text := strings.Repeat("x", 50)
	got, err := Chunk(context.Background(), text, 1, wordCounter)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(got) != 1 || got[0] != text {
		t.Errorf("Chunk() = %v, want the text returned unchanged", got)
	}
}

// A run with nothing to split on — no paragraph break, no sentence
// punctuation, no spaces — that a counter genuinely reports as oversized
// must be accepted as one oversized chunk rather than recursing forever or
// dropping text.
func TestChunkAcceptsAGenuinelyIndivisibleRunAsOneOversizedChunk(t *testing.T) {
	text := strings.Repeat("x", 50)
	got, err := Chunk(context.Background(), text, 10, charCounter)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(got) != 1 || got[0] != text {
		t.Errorf("Chunk() = %v, want the single indivisible run accepted as one oversized chunk despite exceeding maxTokens", got)
	}
}

// pack() must tokenize its separator once per call, not once per piece: in
// production counter is an HTTP round-trip to llama.cpp, so counting the
// separator inside the loop costs an extra request per word for a body that
// falls all the way to word-level splitting.
func TestChunkCountsTheSeparatorOncePerPackCall(t *testing.T) {
	text := strings.Repeat("word ", 500)
	sepCalls := 0
	counting := func(ctx context.Context, s string) (int, error) {
		// only separators are whitespace-only here; splitNonEmpty drops
		// blank pieces, and the full text isn't blank.
		if strings.TrimSpace(s) == "" {
			sepCalls++
		}
		return wordCounter(ctx, s)
	}

	got, err := Chunk(context.Background(), text, 20, counting)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(got) < 2 {
		t.Fatalf("Chunk() = %d chunks, want the long run split into several", len(got))
	}
	// One call per pack() invocation, and pack() recurses only as deep as
	// the fallback tiers go (paragraphs, sentences, words) — a handful,
	// independent of the 500 pieces. Counting per piece instead would be
	// ~475 calls here, one for every piece appended to a non-empty chunk.
	if sepCalls > 5 {
		t.Errorf("counter called %d times for a separator, want one per pack() call (a handful), not one per piece", sepCalls)
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
