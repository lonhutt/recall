// Package embeddings defines the provider-neutral types shared by every
// embedding client implementation.
package embeddings

// InputType distinguishes text being indexed (Document) from text being
// used to query the index (Query): some embedding models encode the two
// differently for better retrieval quality.
type InputType string

const (
	InputTypeQuery    InputType = "query"
	InputTypeDocument InputType = "document"
)
