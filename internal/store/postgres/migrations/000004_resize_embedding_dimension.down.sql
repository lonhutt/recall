ALTER TABLE memories DROP COLUMN embedding;
ALTER TABLE memories ADD COLUMN embedding vector(1024);
CREATE INDEX memories_embedding_idx ON memories USING hnsw (embedding vector_cosine_ops);
