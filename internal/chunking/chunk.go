// Package chunking splits text that's too long for an embedding model's
// context window into pieces that each fit, preferring paragraph
// boundaries, then sentences, then words, so a piece reads as a coherent
// unit rather than an arbitrary token window.
package chunking

import (
	"context"
	"fmt"
	"strings"
)

// TokenCounter counts how many tokens text will produce once embedded,
// e.g. (*llamacpp.Client).CountTokens.
type TokenCounter func(ctx context.Context, text string) (int, error)

// Chunk splits text into pieces that each fit within maxTokens according to
// counter, so every piece can be embedded on its own. Returns a
// single-element slice containing text unchanged when it already fits.
func Chunk(ctx context.Context, text string, maxTokens int, counter TokenCounter) ([]string, error) {
	if maxTokens <= 0 {
		return nil, fmt.Errorf("maxTokens must be positive, got %d", maxTokens)
	}
	n, err := counter(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("counting tokens: %w", err)
	}
	if n <= maxTokens {
		return []string{text}, nil
	}

	paragraphs := splitNonEmpty(text, "\n\n")
	return pack(ctx, paragraphs, "\n\n", maxTokens, counter, splitSentences)
}

// fallbackOrUnchanged applies fallback to piece if given, returning
// []string{piece} unchanged when fallback is nil or found no delimiter to
// split on (so the caller can detect "no progress" uniformly and decide
// whether to try the next fallback tier).
func fallbackOrUnchanged(piece string, fallback func(string) []string) []string {
	if fallback == nil {
		return []string{piece}
	}
	return fallback(piece)
}

// pack greedily groups pieces into chunks joined by sep, each within
// maxTokens. A piece that alone exceeds maxTokens is split further (via
// fallback, or word-splitting if fallback is nil) before being packed.
func pack(ctx context.Context, pieces []string, sep string, maxTokens int, counter TokenCounter, fallback func(string) []string) ([]string, error) {
	var chunks []string
	var current []string
	currentTokens := 0

	flush := func() {
		if len(current) > 0 {
			chunks = append(chunks, strings.Join(current, sep))
			current = nil
			currentTokens = 0
		}
	}

	for _, piece := range pieces {
		pieceTokens, err := counter(ctx, piece)
		if err != nil {
			return nil, fmt.Errorf("counting tokens: %w", err)
		}

		if pieceTokens > maxTokens {
			flush()
			// Try this tier's fallback first; if it finds no delimiter
			// (returns the piece unchanged — e.g. no sentence-ending
			// punctuation), fall through to word-splitting before
			// declaring the piece indivisible. Conflating "this fallback
			// found nothing" with "truly indivisible" would skip
			// word-splitting even when the piece has plenty of spaces to
			// split on.
			subPieces := fallbackOrUnchanged(piece, fallback)
			if len(subPieces) == 1 && subPieces[0] == piece {
				subPieces = splitNonEmpty(piece, " ")
			}
			// A single indivisible run (e.g. one giant word with no
			// spaces) can't be split further even by word; accept it as
			// one oversized chunk rather than recursing forever. Embed
			// will still reject it if it's truly too big to embed — an
			// honest failure, not silent data loss, since nothing here
			// drops text.
			if len(subPieces) == 1 && subPieces[0] == piece {
				chunks = append(chunks, piece)
				continue
			}
			sub, err := pack(ctx, subPieces, " ", maxTokens, counter, nil)
			if err != nil {
				return nil, err
			}
			chunks = append(chunks, sub...)
			continue
		}

		sepTokens := 0
		if len(current) > 0 {
			sepTokens, err = counter(ctx, sep)
			if err != nil {
				return nil, fmt.Errorf("counting tokens: %w", err)
			}
		}
		if len(current) > 0 && currentTokens+sepTokens+pieceTokens > maxTokens {
			flush()
		}
		current = append(current, piece)
		currentTokens += sepTokens + pieceTokens
	}
	flush()
	return chunks, nil
}

func splitNonEmpty(text, sep string) []string {
	var out []string
	for _, p := range strings.Split(text, sep) {
		if strings.TrimSpace(p) != "" {
			out = append(out, p)
		}
	}
	return out
}

// splitSentences breaks text after '.', '!', or '?' followed by whitespace.
// It's a heuristic, not a full sentence tokenizer: worst case it over- or
// under-splits by a few words, and pack() still enforces the token budget
// regardless of where the boundaries land.
func splitSentences(text string) []string {
	var out []string
	start := 0
	for i := 0; i < len(text); i++ {
		c := text[i]
		if (c == '.' || c == '!' || c == '?') && i+1 < len(text) && (text[i+1] == ' ' || text[i+1] == '\n') {
			if piece := strings.TrimSpace(text[start : i+1]); piece != "" {
				out = append(out, piece)
			}
			start = i + 1
		}
	}
	if rest := strings.TrimSpace(text[start:]); rest != "" {
		out = append(out, rest)
	}
	return out
}
