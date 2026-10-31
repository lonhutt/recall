ALTER TABLE memories ADD COLUMN superseded_by uuid REFERENCES memories(id) ON DELETE SET NULL;
ALTER TABLE memories ADD COLUMN overrides uuid REFERENCES memories(id) ON DELETE SET NULL;

CREATE INDEX memories_superseded_by_idx ON memories (superseded_by) WHERE superseded_by IS NOT NULL;
CREATE INDEX memories_overrides_idx ON memories (overrides) WHERE overrides IS NOT NULL;
