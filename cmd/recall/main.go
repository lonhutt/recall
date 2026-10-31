// Command recall runs Recall's MCP server, applies its database migrations,
// or scores search ranking against a probe file.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/lonhutt/recall/internal/config"
	"github.com/lonhutt/recall/internal/embeddings/llamacpp"
	"github.com/lonhutt/recall/internal/httpauth"
	"github.com/lonhutt/recall/internal/store/postgres"
	"github.com/lonhutt/recall/internal/tools"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	cmd := "serve"
	if len(os.Args) > 1 {
		cmd = os.Args[1]
	}

	switch cmd {
	case "migrate":
		dbURL := os.Getenv("RECALL_DATABASE_URL")
		if dbURL == "" {
			return fmt.Errorf("missing required environment variable: RECALL_DATABASE_URL")
		}
		return postgres.RunMigrations(dbURL)
	case "backfill-chunks":
		cfg, err := config.Load()
		if err != nil {
			return err
		}
		return backfillChunks(cfg)
	case "eval":
		cfg, err := config.Load()
		if err != nil {
			return err
		}
		return runEval(cfg, os.Stdin, os.Stdout)
	case "serve":
		cfg, err := config.Load()
		if err != nil {
			return err
		}
		return serve(cfg)
	default:
		return fmt.Errorf("unknown command %q; expected \"serve\", \"migrate\", \"backfill-chunks\", or \"eval\"", cmd)
	}
}

// backfillChunks re-embeds every existing memory's description and body
// chunks under the current chunked-search scheme. Safe to run more than
// once: each run fully replaces the previous chunk set.
//
// Each memory's update is its own transaction, so running this won't
// corrupt data or block serving — but a memory created concurrently during
// the run can shift pagination and cause one untouched row to be skipped,
// so prefer running it during a quiet window, and re-run it (safe, since
// it's idempotent) if you suspect stragglers.
func backfillChunks(cfg *config.Config) error {
	ctx := context.Background()

	store, err := postgres.New(ctx, cfg.DatabaseURL)
	if err != nil {
		return err
	}
	defer store.Close()

	embedder := llamacpp.New(cfg.EmbeddingBaseURL, cfg.EmbeddingModel, cfg.EmbeddingQueryPrefix, cfg.EmbeddingDocumentPrefix, cfg.EmbeddingMaxTokens)
	t := &tools.Tools{
		Store:             store,
		Embedder:          embedder,
		EmbeddingModel:    cfg.EmbeddingModel,
		MaxTokensPerChunk: cfg.EmbeddingMaxTokens,
		DocumentPrefix:    cfg.EmbeddingDocumentPrefix,
	}

	const pageSize = 50
	offset := 0
	for {
		memories, err := store.ListMemories(ctx, postgres.MemoryFilter{}, pageSize, offset)
		if err != nil {
			return fmt.Errorf("listing memories: %w", err)
		}
		if len(memories) == 0 {
			break
		}
		for _, m := range memories {
			if err := t.BackfillMemoryEmbedding(ctx, m.Slug, m.Description, m.Body); err != nil {
				return fmt.Errorf("backfilling %q: %w", m.Slug, err)
			}
			slog.Info("backfilled memory chunks", "slug", m.Slug)
		}
		offset += len(memories)
	}
	return nil
}

func serve(cfg *config.Config) error {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	if cfg.MigrateOnStart {
		if err := postgres.RunMigrations(cfg.DatabaseURL); err != nil {
			return fmt.Errorf("running migrations: %w", err)
		}
	}

	store, err := postgres.New(ctx, cfg.DatabaseURL)
	if err != nil {
		return err
	}
	defer store.Close()

	embedder := llamacpp.New(cfg.EmbeddingBaseURL, cfg.EmbeddingModel, cfg.EmbeddingQueryPrefix, cfg.EmbeddingDocumentPrefix, cfg.EmbeddingMaxTokens)

	t := &tools.Tools{
		Store:             store,
		Embedder:          embedder,
		EmbeddingModel:    cfg.EmbeddingModel,
		MaxTokensPerChunk: cfg.EmbeddingMaxTokens,
		DocumentPrefix:    cfg.EmbeddingDocumentPrefix,
	}
	server := mcp.NewServer(&mcp.Implementation{Name: "recall", Version: "0.1.0"}, nil)
	t.RegisterAll(server)

	if cfg.Transport == "http" {
		return serveHTTP(ctx, cfg, server)
	}

	slog.Info("recall MCP server starting", "transport", "stdio")
	return server.Run(ctx, &mcp.StdioTransport{})
}

func serveHTTP(ctx context.Context, cfg *config.Config, server *mcp.Server) error {
	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, nil)

	mux := http.NewServeMux()
	mux.Handle("/mcp", httpauth.Bearer(cfg.HTTPToken, handler))
	// liveness only, so it stays unauthenticated; it reports nothing about the
	// database or the embeddings server, just that this process is serving.
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("ok"))
	})

	httpServer := &http.Server{Addr: ":" + cfg.HTTPPort, Handler: mux}
	go func() {
		<-ctx.Done()
		// give in-flight tool calls a moment to answer; a write that already
		// committed shouldn't look like a network failure to the client. The
		// timeout is there because MCP holds long-lived streams open and
		// Shutdown otherwise waits on them forever.
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(shutdownCtx); err != nil {
			httpServer.Close()
		}
	}()

	slog.Info("recall MCP server starting", "transport", "http", "port", cfg.HTTPPort)
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return err
	}
	return nil
}
