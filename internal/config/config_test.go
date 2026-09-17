package config

import (
	"strings"
	"testing"
)

func withEnv(t *testing.T, env map[string]string) {
	t.Helper()
	for _, k := range []string{
		"RECALL_DATABASE_URL", "RECALL_EMBEDDING_BASE_URL", "RECALL_EMBEDDING_MODEL",
		"RECALL_EMBEDDING_QUERY_PREFIX", "RECALL_EMBEDDING_DOCUMENT_PREFIX",
		"RECALL_MIGRATE_ON_START", "RECALL_TRANSPORT", "RECALL_HTTP_PORT", "RECALL_HTTP_TOKEN",
	} {
		t.Setenv(k, env[k])
	}
}

func TestLoadReturnsConfigWhenRequiredVarsSet(t *testing.T) {
	withEnv(t, map[string]string{
		"RECALL_DATABASE_URL": "postgres://localhost/recall",
	})

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.DatabaseURL != "postgres://localhost/recall" {
		t.Errorf("DatabaseURL = %q", cfg.DatabaseURL)
	}
	if cfg.EmbeddingBaseURL != "http://127.0.0.1:8091" {
		t.Errorf("EmbeddingBaseURL = %q, want default http://127.0.0.1:8091", cfg.EmbeddingBaseURL)
	}
	if cfg.EmbeddingModel != "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF" {
		t.Errorf("EmbeddingModel = %q, want default embeddinggemma repo", cfg.EmbeddingModel)
	}
	if cfg.EmbeddingQueryPrefix != "task: search result | query: " {
		t.Errorf("EmbeddingQueryPrefix = %q, want EmbeddingGemma's default query prefix", cfg.EmbeddingQueryPrefix)
	}
	if cfg.EmbeddingDocumentPrefix != "title: none | text: " {
		t.Errorf("EmbeddingDocumentPrefix = %q, want EmbeddingGemma's default document prefix", cfg.EmbeddingDocumentPrefix)
	}
	if !cfg.MigrateOnStart {
		t.Errorf("MigrateOnStart = false, want default true")
	}
	if cfg.Transport != "stdio" {
		t.Errorf("Transport = %q, want default stdio", cfg.Transport)
	}
	if cfg.HTTPPort != "8092" {
		t.Errorf("HTTPPort = %q, want default 8092", cfg.HTTPPort)
	}
}

func TestLoadReportsMissingRequiredVars(t *testing.T) {
	withEnv(t, map[string]string{})

	_, err := Load()
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
	if !strings.Contains(err.Error(), "RECALL_DATABASE_URL") {
		t.Errorf("error = %q, want it to name the missing var", err.Error())
	}
}

func TestLoadHonorsEmbeddingOverrides(t *testing.T) {
	withEnv(t, map[string]string{
		"RECALL_DATABASE_URL":              "postgres://localhost/recall",
		"RECALL_EMBEDDING_BASE_URL":        "http://localhost:9999",
		"RECALL_EMBEDDING_MODEL":           "custom-model",
		"RECALL_EMBEDDING_QUERY_PREFIX":    "search_query: ",
		"RECALL_EMBEDDING_DOCUMENT_PREFIX": "search_document: ",
	})

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.EmbeddingBaseURL != "http://localhost:9999" {
		t.Errorf("EmbeddingBaseURL = %q, want http://localhost:9999", cfg.EmbeddingBaseURL)
	}
	if cfg.EmbeddingModel != "custom-model" {
		t.Errorf("EmbeddingModel = %q, want custom-model", cfg.EmbeddingModel)
	}
	if cfg.EmbeddingQueryPrefix != "search_query: " {
		t.Errorf("EmbeddingQueryPrefix = %q, want search_query: ", cfg.EmbeddingQueryPrefix)
	}
	if cfg.EmbeddingDocumentPrefix != "search_document: " {
		t.Errorf("EmbeddingDocumentPrefix = %q, want search_document: ", cfg.EmbeddingDocumentPrefix)
	}
}

func TestLoadRejectsInvalidMigrateOnStart(t *testing.T) {
	withEnv(t, map[string]string{
		"RECALL_DATABASE_URL":     "postgres://localhost/recall",
		"RECALL_MIGRATE_ON_START": "not-a-bool",
	})

	if _, err := Load(); err == nil {
		t.Fatal("expected an error for an invalid RECALL_MIGRATE_ON_START, got nil")
	}
}

func TestLoadRejectsInvalidTransport(t *testing.T) {
	withEnv(t, map[string]string{
		"RECALL_DATABASE_URL": "postgres://localhost/recall",
		"RECALL_TRANSPORT":    "carrier-pigeon",
	})

	if _, err := Load(); err == nil {
		t.Fatal("expected an error for an invalid RECALL_TRANSPORT, got nil")
	}
}

func TestLoadRejectsHTTPTransportWithoutToken(t *testing.T) {
	withEnv(t, map[string]string{
		"RECALL_DATABASE_URL": "postgres://localhost/recall",
		"RECALL_TRANSPORT":    "http",
	})

	_, err := Load()
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
	if !strings.Contains(err.Error(), "RECALL_HTTP_TOKEN") {
		t.Errorf("error = %q, want it to mention RECALL_HTTP_TOKEN", err.Error())
	}
}

func TestLoadAcceptsHTTPTransportWithToken(t *testing.T) {
	withEnv(t, map[string]string{
		"RECALL_DATABASE_URL": "postgres://localhost/recall",
		"RECALL_TRANSPORT":    "http",
		"RECALL_HTTP_PORT":    "9999",
		"RECALL_HTTP_TOKEN":   "secret",
	})

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.HTTPPort != "9999" {
		t.Errorf("HTTPPort = %q, want 9999", cfg.HTTPPort)
	}
	if cfg.HTTPToken != "secret" {
		t.Errorf("HTTPToken = %q, want secret", cfg.HTTPToken)
	}
}
