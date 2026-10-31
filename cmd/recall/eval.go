package main

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/lonhutt/recall/internal/config"
	"github.com/lonhutt/recall/internal/embeddings"
	"github.com/lonhutt/recall/internal/embeddings/llamacpp"
	"github.com/lonhutt/recall/internal/eval"
	"github.com/lonhutt/recall/internal/models"
	"github.com/lonhutt/recall/internal/store/postgres"
)

// runEval compares vector and hybrid search ranking on the probes read from in.
func runEval(cfg *config.Config, in io.Reader, out io.Writer) error {
	ctx := context.Background()

	probes, err := eval.LoadProbes(in)
	if err != nil {
		return err
	}

	store, err := postgres.New(ctx, cfg.DatabaseURL)
	if err != nil {
		return err
	}
	defer store.Close()

	embedder := llamacpp.New(cfg.EmbeddingBaseURL, cfg.EmbeddingModel, cfg.EmbeddingQueryPrefix, cfg.EmbeddingDocumentPrefix, cfg.EmbeddingMaxTokens)

	unknown, err := eval.UnknownWants(probes, func(slug string) (bool, error) {
		_, err := store.GetMemoryBySlug(ctx, slug)
		if errors.Is(err, models.ErrNotFound) {
			return false, nil
		}
		return err == nil, err
	})
	if err != nil {
		return err
	}
	for _, slug := range unknown {
		fmt.Fprintf(out, "warning: no memory with slug %q\n", slug)
	}

	// Embedded once up front so both modes rank the exact same vector.
	vectors := make(map[string][]float32, len(probes))
	for _, p := range probes {
		if _, ok := vectors[p.Query]; ok {
			continue
		}
		v, err := embedder.Embed(ctx, []string{p.Query}, embeddings.InputTypeQuery)
		if err != nil {
			return fmt.Errorf("embedding %q: %w", p.Query, err)
		}
		vectors[p.Query] = v[0]
	}

	// searcher with keyword=false passes an empty query, which is exactly
	// what search_memory did before hybrid search: the baseline to beat.
	searcher := func(keyword bool) eval.Searcher {
		return func(ctx context.Context, p eval.Probe) ([]string, error) {
			filter := postgres.MemoryFilter{EmbeddingModel: &cfg.EmbeddingModel}
			if p.Project != "" {
				filter.Project = &p.Project
			}
			query := ""
			if keyword {
				query = p.Query
			}
			scored, err := store.SearchMemories(ctx, vectors[p.Query], query, filter, 10)
			if err != nil {
				return nil, err
			}
			slugs := make([]string, len(scored))
			for i, s := range scored {
				slugs[i] = s.Slug
			}
			return slugs, nil
		}
	}

	var modes []eval.Mode
	for _, m := range []struct {
		name    string
		keyword bool
	}{{"vector", false}, {"hybrid", true}} {
		outcomes, err := eval.Run(ctx, probes, searcher(m.keyword))
		if err != nil {
			return fmt.Errorf("%s: %w", m.name, err)
		}
		modes = append(modes, eval.Mode{Name: m.name, Outcomes: outcomes})
	}
	return eval.Report(out, modes)
}
