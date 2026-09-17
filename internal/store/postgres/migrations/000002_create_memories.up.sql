CREATE TABLE memories (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    type            text NOT NULL,
    slug            text NOT NULL,
    description     text NOT NULL,
    body            text NOT NULL,
    project         text,
    agent           text,
    tags            text[] NOT NULL DEFAULT '{}',
    embedding       vector(1024) NOT NULL,
    embedding_model text NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX memories_slug_key      ON memories (slug);
CREATE INDEX        memories_type_idx      ON memories (type);
CREATE INDEX        memories_project_idx   ON memories (project) WHERE project IS NOT NULL;
CREATE INDEX        memories_agent_idx     ON memories (agent) WHERE agent IS NOT NULL;
CREATE INDEX        memories_tags_idx      ON memories USING gin (tags);
CREATE INDEX        memories_embedding_idx ON memories USING hnsw (embedding vector_cosine_ops);
