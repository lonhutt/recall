// Package llamacpp is an HTTP client for a local llama.cpp server's
// OpenAI-compatible embeddings endpoint.
package llamacpp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/lonhutt/recall/internal/embeddings"
)

const embeddingsPath = "/v1/embeddings"

// DefaultQueryPrefix and DefaultDocumentPrefix are EmbeddingGemma's
// documented asymmetric-retrieval prompt format: queries and documents are
// encoded differently for better retrieval
// (https://ai.google.dev/gemma/docs/embeddinggemma/model_card). A different
// model family likely needs different prefixes (or none) — pass overrides
// to New rather than editing these.
const (
	DefaultQueryPrefix    = "task: search result | query: "
	DefaultDocumentPrefix = "title: none | text: "
)

var defaultBackoff = []time.Duration{250 * time.Millisecond, 750 * time.Millisecond}

type Client struct {
	baseURL        string
	model          string
	queryPrefix    string
	documentPrefix string
	httpClient     *http.Client
	backoff        []time.Duration
}

func New(baseURL, model, queryPrefix, documentPrefix string) *Client {
	return &Client{
		baseURL:        baseURL,
		model:          model,
		queryPrefix:    queryPrefix,
		documentPrefix: documentPrefix,
		httpClient:     http.DefaultClient,
		backoff:        defaultBackoff,
	}
}

// APIError wraps a non-2xx response from the llama.cpp server.
type APIError struct {
	StatusCode int
	Detail     string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("llama.cpp server returned %d: %s", e.StatusCode, e.Detail)
}

type embeddingRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

type embeddingDatum struct {
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

type embeddingResponse struct {
	Data []embeddingDatum `json:"data"`
}

func (c *Client) prefix(text string, kind embeddings.InputType) string {
	if kind == embeddings.InputTypeQuery {
		return c.queryPrefix + text
	}
	return c.documentPrefix + text
}

// Embed returns one vector per input text, in the same order as texts.
func (c *Client) Embed(ctx context.Context, texts []string, kind embeddings.InputType) ([][]float32, error) {
	prefixed := make([]string, len(texts))
	for i, t := range texts {
		prefixed[i] = c.prefix(t, kind)
	}

	body, err := json.Marshal(embeddingRequest{Input: prefixed, Model: c.model})
	if err != nil {
		return nil, fmt.Errorf("marshaling llama.cpp request: %w", err)
	}

	var lastErr error
	for attempt := 0; ; attempt++ {
		vectors, retriable, err := c.doEmbed(ctx, body)
		if err == nil {
			return vectors, nil
		}
		lastErr = err
		if !retriable || attempt >= len(c.backoff) {
			return nil, lastErr
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(c.backoff[attempt]):
		}
	}
}

func (c *Client) doEmbed(ctx context.Context, body []byte) ([][]float32, bool, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+embeddingsPath, bytes.NewReader(body))
	if err != nil {
		return nil, false, fmt.Errorf("building llama.cpp request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, true, fmt.Errorf("calling llama.cpp server: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, true, fmt.Errorf("reading llama.cpp response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var errBody struct {
			Detail string `json:"detail"`
		}
		_ = json.Unmarshal(respBody, &errBody)
		apiErr := &APIError{StatusCode: resp.StatusCode, Detail: errBody.Detail}
		return nil, resp.StatusCode >= 500, apiErr
	}

	var parsed embeddingResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, false, fmt.Errorf("decoding llama.cpp response: %w", err)
	}

	vectors := make([][]float32, len(parsed.Data))
	for _, d := range parsed.Data {
		vectors[d.Index] = d.Embedding
	}
	return vectors, false, nil
}
