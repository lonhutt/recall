ALTER TABLE memories ALTER COLUMN embedding SET NOT NULL;
CREATE INDEX memories_embedding_idx ON memories USING hnsw (embedding vector_cosine_ops);
