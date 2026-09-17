DROP INDEX IF EXISTS memories_embedding_idx;
ALTER TABLE memories ALTER COLUMN embedding DROP NOT NULL;
