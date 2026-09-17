package llamacpp

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/lonhutt/recall/internal/embeddings"
)

func TestEmbedPrefixesQueryTextAndParsesResponse(t *testing.T) {
	var gotPath string
	var gotBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decoding request body: %v", err)
		}
		resp := map[string]any{
			"data": []map[string]any{
				{"index": 1, "embedding": []float32{0.4, 0.5}},
				{"index": 0, "embedding": []float32{0.1, 0.2}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	c := New(srv.URL, "embeddinggemma", DefaultQueryPrefix, DefaultDocumentPrefix)

	got, err := c.Embed(context.Background(), []string{"first", "second"}, embeddings.InputTypeQuery)
	if err != nil {
		t.Fatalf("Embed returned error: %v", err)
	}

	if gotPath != "/v1/embeddings" {
		t.Errorf("path = %q, want /v1/embeddings", gotPath)
	}
	input, ok := gotBody["input"].([]any)
	if !ok || len(input) != 2 {
		t.Fatalf("body[input] = %v", gotBody["input"])
	}
	if input[0] != "task: search result | query: first" {
		t.Errorf("input[0] = %q, want query-prefixed text", input[0])
	}
	if gotBody["model"] != "embeddinggemma" {
		t.Errorf("body[model] = %v, want embeddinggemma", gotBody["model"])
	}

	want := [][]float32{{0.1, 0.2}, {0.4, 0.5}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Embed() = %v, want %v (must be re-ordered by index)", got, want)
	}
}

func TestEmbedPrefixesDocumentText(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"index": 0, "embedding": []float32{0.1}}},
		})
	}))
	defer srv.Close()

	c := New(srv.URL, "embeddinggemma", DefaultQueryPrefix, DefaultDocumentPrefix)
	if _, err := c.Embed(context.Background(), []string{"hello"}, embeddings.InputTypeDocument); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	input := gotBody["input"].([]any)
	if input[0] != "title: none | text: hello" {
		t.Errorf("input[0] = %q, want document-prefixed text", input[0])
	}
}

func TestEmbedUsesCustomPrefixesWhenConfigured(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"index": 0, "embedding": []float32{0.1}}},
		})
	}))
	defer srv.Close()

	c := New(srv.URL, "some-other-model", "search_query: ", "search_document: ")
	if _, err := c.Embed(context.Background(), []string{"hello"}, embeddings.InputTypeQuery); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	input := gotBody["input"].([]any)
	if input[0] != "search_query: hello" {
		t.Errorf("input[0] = %q, want the configured query prefix applied", input[0])
	}
}

func TestEmbedMapsNonSuccessResponseToAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"detail": "model not loaded"})
	}))
	defer srv.Close()

	c := New(srv.URL, "embeddinggemma", DefaultQueryPrefix, DefaultDocumentPrefix)
	c.backoff = nil

	_, err := c.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery)
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error is not *APIError: %v", err)
	}
	if apiErr.StatusCode != http.StatusInternalServerError {
		t.Errorf("StatusCode = %d, want 500", apiErr.StatusCode)
	}
	if apiErr.Detail != "model not loaded" {
		t.Errorf("Detail = %q, want %q", apiErr.Detail, "model not loaded")
	}
}

func TestEmbedRetriesOnServerErrorThenSucceeds(t *testing.T) {
	attempts := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			json.NewEncoder(w).Encode(map[string]string{"detail": "loading model"})
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"index": 0, "embedding": []float32{0.9}}},
		})
	}))
	defer srv.Close()

	c := New(srv.URL, "embeddinggemma", DefaultQueryPrefix, DefaultDocumentPrefix)
	c.backoff = []time.Duration{time.Millisecond, time.Millisecond}

	got, err := c.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery)
	if err != nil {
		t.Fatalf("Embed returned error after retry: %v", err)
	}
	if attempts != 2 {
		t.Errorf("attempts = %d, want 2", attempts)
	}
	want := [][]float32{{0.9}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Embed() = %v, want %v", got, want)
	}
}
