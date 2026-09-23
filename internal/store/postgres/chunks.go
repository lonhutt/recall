package postgres

import (
	"context"

	"github.com/jackc/pgx/v5"
	pgvector "github.com/pgvector/pgvector-go"
)

// MemoryChunk is one piece of a memory's body embedded on its own, for a
// body too long to fit the embedding model's context window in a single
// vector. ChunkIndex is 0-based and defines chunk order; a plain memory
// with a short body still gets exactly one chunk.
type MemoryChunk struct {
	ChunkIndex int
	Text       string
	Embedding  []float32
}

// replaceMemoryChunks deletes memoryID's existing chunks and inserts chunks
// in their place, as part of tx: a chunk set is always replaced wholesale,
// never patched, so an update can never leave a stale chunk from the
// previous body version sitting alongside the new ones.
func replaceMemoryChunks(ctx context.Context, tx pgx.Tx, memoryID string, chunks []MemoryChunk) error {
	if _, err := tx.Exec(ctx, `DELETE FROM memory_chunks WHERE memory_id = $1`, memoryID); err != nil {
		return err
	}
	for _, c := range chunks {
		if _, err := tx.Exec(ctx,
			`INSERT INTO memory_chunks (memory_id, chunk_index, chunk_text, embedding) VALUES ($1, $2, $3, $4)`,
			memoryID, c.ChunkIndex, c.Text, pgvector.NewVector(c.Embedding),
		); err != nil {
			return err
		}
	}
	return nil
}

// CountMemoryChunks returns how many chunk rows exist for slug's memory.
// Mainly for tests and diagnostics — production code only ever consumes
// chunks indirectly, through what SearchMemories returns.
func (s *Store) CountMemoryChunks(ctx context.Context, slug string) (int, error) {
	var count int
	err := s.pool.QueryRow(ctx,
		`SELECT count(*) FROM memory_chunks mc JOIN memories m ON m.id = mc.memory_id WHERE m.slug = $1`,
		slug,
	).Scan(&count)
	return count, err
}
