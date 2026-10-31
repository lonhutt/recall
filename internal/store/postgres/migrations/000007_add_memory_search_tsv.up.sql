-- body is capped because a tsvector over 1MB is an error, which would fail the
-- whole save; the vector chunks still cover the rest of a long body.
ALTER TABLE memories ADD COLUMN search_tsv tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('english', slug), 'A') ||
    setweight(to_tsvector('english', description), 'A') ||
    setweight(to_tsvector('english', left(body, 100000)), 'B')
) STORED;
CREATE INDEX memories_search_tsv_idx ON memories USING gin (search_tsv);
