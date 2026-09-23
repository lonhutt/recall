package llamacpp

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/lonhutt/recall/internal/embeddings"
)

// The EmbeddingGemma prefix pair, duplicated here so these tests run against
// a realistic one; internal/config owns the production defaults.
const (
	gemmaQueryPrefix    = "task: search result | query: "
	gemmaDocumentPrefix = "title: none | text: "
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

	c := New(srv.URL, "embeddinggemma", gemmaQueryPrefix, gemmaDocumentPrefix, 0)

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

	c := New(srv.URL, "embeddinggemma", gemmaQueryPrefix, gemmaDocumentPrefix, 0)
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

	c := New(srv.URL, "some-other-model", "search_query: ", "search_document: ", 0)
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

	c := New(srv.URL, "embeddinggemma", gemmaQueryPrefix, gemmaDocumentPrefix, 0)
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

	c := New(srv.URL, "embeddinggemma", gemmaQueryPrefix, gemmaDocumentPrefix, 0)
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

// the next three cover a malformed response from the embeddings server;
// callers index into it by position (vectors[0]), so a bad response used to
// panic instead of erroring.
func TestEmbedRejectsOutOfRangeIndex(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"index": 1, "embedding": []float32{0.1}}},
		})
	}))
	defer srv.Close()

	c := New(srv.URL, "embeddinggemma", gemmaQueryPrefix, gemmaDocumentPrefix, 0)
	if _, err := c.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery); err == nil {
		t.Fatal("expected an error for an out-of-range index, got nil")
	}
}

func TestEmbedRejectsWrongEmbeddingCount(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"data": []map[string]any{}})
	}))
	defer srv.Close()

	c := New(srv.URL, "embeddinggemma", gemmaQueryPrefix, gemmaDocumentPrefix, 0)
	if _, err := c.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery); err == nil {
		t.Fatal("expected an error for zero embeddings, got nil")
	}
}

func TestEmbedRejectsDuplicateIndex(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"index": 0, "embedding": []float32{0.1}},
				{"index": 0, "embedding": []float32{0.2}},
			},
		})
	}))
	defer srv.Close()

	c := New(srv.URL, "embeddinggemma", gemmaQueryPrefix, gemmaDocumentPrefix, 0)
	if _, err := c.Embed(context.Background(), []string{"a", "b"}, embeddings.InputTypeQuery); err == nil {
		t.Fatal("expected an error when one input got no embedding, got nil")
	}
}

func TestCountTokensParsesResponse(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenize" {
			t.Errorf("path = %q, want /tokenize", r.URL.Path)
		}
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(map[string]any{"tokens": []int{1, 2, 3, 4, 5}})
	}))
	defer srv.Close()

	c := New(srv.URL, "embeddinggemma", gemmaQueryPrefix, gemmaDocumentPrefix, 0)
	n, err := c.CountTokens(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	if n != 5 {
		t.Errorf("CountTokens() = %d, want 5", n)
	}
	if gotBody["content"] != "hello world" {
		t.Errorf("body[content] = %v, want %q", gotBody["content"], "hello world")
	}
}

// a document that has grown past the embedding model's context window is
// exactly the failure this guards against: without it, the caller only
// finds out from an opaque error at the embeddings endpoint (or, on some
// servers, silent truncation) instead of a message that says what's wrong.
func TestEmbedRejectsTextExceedingMaxTokens(t *testing.T) {
	embeddingsCalled := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/tokenize":
			json.NewEncoder(w).Encode(map[string]any{"tokens": make([]int, 2372)})
		case "/v1/embeddings":
			embeddingsCalled = true
			json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"index": 0, "embedding": []float32{0.1}}},
			})
		}
	}))
	defer srv.Close()

	c := New(srv.URL, "embeddinggemma", gemmaQueryPrefix, gemmaDocumentPrefix, 2048)
	_, err := c.Embed(context.Background(), []string{"a very long memory body"}, embeddings.InputTypeDocument)
	if err == nil {
		t.Fatal("expected an error for text exceeding the token limit, got nil")
	}
	if !strings.Contains(err.Error(), "2372") || !strings.Contains(err.Error(), "2048") {
		t.Errorf("error = %q, want it to name the actual token count and the limit", err.Error())
	}
	if embeddingsCalled {
		t.Error("Embed called the embeddings endpoint despite the text exceeding the token limit")
	}
}

func TestEmbedAllowsTextWithinMaxTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/tokenize":
			json.NewEncoder(w).Encode(map[string]any{"tokens": make([]int, 100)})
		case "/v1/embeddings":
			json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"index": 0, "embedding": []float32{0.1}}},
			})
		}
	}))
	defer srv.Close()

	c := New(srv.URL, "embeddinggemma", gemmaQueryPrefix, gemmaDocumentPrefix, 2048)
	if _, err := c.Embed(context.Background(), []string{"a short memory body"}, embeddings.InputTypeDocument); err != nil {
		t.Fatalf("Embed: %v", err)
	}
}

func TestEmbedSkipsTokenCheckWhenMaxTokensIsZero(t *testing.T) {
	tokenizeCalled := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/tokenize":
			tokenizeCalled = true
			json.NewEncoder(w).Encode(map[string]any{"tokens": make([]int, 999999)})
		case "/v1/embeddings":
			json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"index": 0, "embedding": []float32{0.1}}},
			})
		}
	}))
	defer srv.Close()

	c := New(srv.URL, "embeddinggemma", gemmaQueryPrefix, gemmaDocumentPrefix, 0)
	if _, err := c.Embed(context.Background(), []string{"anything"}, embeddings.InputTypeDocument); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if tokenizeCalled {
		t.Error("Embed called /tokenize despite maxTokens being 0 (disabled)")
	}
}
