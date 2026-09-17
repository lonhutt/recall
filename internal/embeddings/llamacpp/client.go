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

// requestTimeout bounds a single embed call. The case this is for is an
// embeddings server that accepts the connection and then never answers
// (llama.cpp still loading the model); the retry ladder below can't help
// there, since nothing ever comes back to retry.
const requestTimeout = 30 * time.Second

var defaultBackoff = []time.Duration{250 * time.Millisecond, 750 * time.Millisecond}

type Client struct {
	baseURL        string
	model          string
	queryPrefix    string
	documentPrefix string
	httpClient     *http.Client
	backoff        []time.Duration
}

// New builds a client for the llama.cpp server at baseURL. The prefixes
// implement the embedding model's asymmetric-retrieval convention, if the
// model has one; internal/config holds the defaults.
func New(baseURL, model, queryPrefix, documentPrefix string) *Client {
	return &Client{
		baseURL:        baseURL,
		model:          model,
		queryPrefix:    queryPrefix,
		documentPrefix: documentPrefix,
		httpClient:     &http.Client{Timeout: requestTimeout},
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

// Embed returns exactly one non-empty vector per input text, in the same
// order as texts, or an error; callers can index the result by input
// position.
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
		vectors, retriable, err := c.doEmbed(ctx, body, len(texts))
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

func (c *Client) doEmbed(ctx context.Context, body []byte, want int) ([][]float32, bool, error) {
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

	// the response is untrusted; a bad count, a duplicate or out-of-range
	// index, or an empty vector would otherwise reach callers as a short
	// slice or a nil element, and they index into it directly.
	if len(parsed.Data) != want {
		return nil, false, fmt.Errorf("llama.cpp returned %d embeddings for %d inputs", len(parsed.Data), want)
	}
	vectors := make([][]float32, want)
	for _, d := range parsed.Data {
		if d.Index < 0 || d.Index >= want {
			return nil, false, fmt.Errorf("llama.cpp returned out-of-range embedding index %d for %d inputs", d.Index, want)
		}
		if len(d.Embedding) == 0 {
			return nil, false, fmt.Errorf("llama.cpp returned an empty embedding at index %d", d.Index)
		}
		vectors[d.Index] = d.Embedding
	}
	for i, v := range vectors {
		if v == nil {
			return nil, false, fmt.Errorf("llama.cpp returned no embedding for input %d", i)
		}
	}
	return vectors, false, nil
}
