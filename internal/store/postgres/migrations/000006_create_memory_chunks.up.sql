CREATE TABLE memory_chunks (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id   uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    chunk_index int NOT NULL,
    chunk_text  text NOT NULL,
    embedding   vector(768) NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (memory_id, chunk_index)
);

CREATE INDEX memory_chunks_memory_id_idx ON memory_chunks (memory_id);
CREATE INDEX memory_chunks_embedding_idx ON memory_chunks USING hnsw (embedding vector_cosine_ops);
