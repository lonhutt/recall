DROP INDEX IF EXISTS memories_embedding_idx;
ALTER TABLE memories DROP COLUMN embedding;
ALTER TABLE memories ADD COLUMN embedding vector(768);
