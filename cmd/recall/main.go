// Command recall runs Recall's MCP server, or applies its database migrations.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"

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
	case "serve":
		cfg, err := config.Load()
		if err != nil {
			return err
		}
		return serve(cfg)
	default:
		return fmt.Errorf("unknown command %q; expected \"serve\" or \"migrate\"", cmd)
	}
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

	embedder := llamacpp.New(cfg.EmbeddingBaseURL, cfg.EmbeddingModel, cfg.EmbeddingQueryPrefix, cfg.EmbeddingDocumentPrefix)

	t := &tools.Tools{Store: store, Embedder: embedder, EmbeddingModel: cfg.EmbeddingModel}
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

	httpServer := &http.Server{Addr: ":" + cfg.HTTPPort, Handler: mux}
	go func() {
		<-ctx.Done()
		httpServer.Close()
	}()

	slog.Info("recall MCP server starting", "transport", "http", "port", cfg.HTTPPort)
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return err
	}
	return nil
}
