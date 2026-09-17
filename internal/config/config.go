// Package config loads Recall's runtime configuration from environment variables.
package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

type Config struct {
	DatabaseURL             string
	EmbeddingBaseURL        string
	EmbeddingModel          string
	EmbeddingQueryPrefix    string
	EmbeddingDocumentPrefix string
	MigrateOnStart          bool
	Transport               string
	HTTPPort                string
	HTTPToken               string
}

func Load() (*Config, error) {
	cfg := &Config{
		DatabaseURL:      os.Getenv("RECALL_DATABASE_URL"),
		EmbeddingBaseURL: envOrDefault("RECALL_EMBEDDING_BASE_URL", "http://127.0.0.1:8091"),
		EmbeddingModel:   envOrDefault("RECALL_EMBEDDING_MODEL", "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF"),
		// Defaults match EmbeddingGemma's documented asymmetric-retrieval prompt
		// convention (see internal/embeddings/llamacpp). Override these if the
		// embedding model changes to one with a different (or no) convention.
		EmbeddingQueryPrefix:    envOrDefault("RECALL_EMBEDDING_QUERY_PREFIX", "task: search result | query: "),
		EmbeddingDocumentPrefix: envOrDefault("RECALL_EMBEDDING_DOCUMENT_PREFIX", "title: none | text: "),
		Transport:               envOrDefault("RECALL_TRANSPORT", "stdio"),
		HTTPPort:                envOrDefault("RECALL_HTTP_PORT", "8092"),
		HTTPToken:               os.Getenv("RECALL_HTTP_TOKEN"),
	}

	var missing []string
	if cfg.DatabaseURL == "" {
		missing = append(missing, "RECALL_DATABASE_URL")
	}
	if len(missing) > 0 {
		return nil, fmt.Errorf("missing required environment variables: %s", strings.Join(missing, ", "))
	}

	if cfg.Transport != "stdio" && cfg.Transport != "http" {
		return nil, fmt.Errorf("invalid RECALL_TRANSPORT %q; expected \"stdio\" or \"http\"", cfg.Transport)
	}
	if cfg.Transport == "http" && cfg.HTTPToken == "" {
		return nil, fmt.Errorf("RECALL_HTTP_TOKEN is required when RECALL_TRANSPORT=http")
	}

	migrateOnStart, err := strconv.ParseBool(envOrDefault("RECALL_MIGRATE_ON_START", "true"))
	if err != nil {
		return nil, fmt.Errorf("parsing RECALL_MIGRATE_ON_START: %w", err)
	}
	cfg.MigrateOnStart = migrateOnStart

	return cfg, nil
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
