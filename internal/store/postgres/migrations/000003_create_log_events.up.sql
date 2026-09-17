CREATE TABLE log_events (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    occurred_at timestamptz NOT NULL DEFAULT now(),
    description text NOT NULL,
    project     text,
    agent       text,
    tags        text[] NOT NULL DEFAULT '{}',
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX log_events_occurred_at_idx ON log_events (occurred_at DESC);
CREATE INDEX log_events_project_idx     ON log_events (project) WHERE project IS NOT NULL;
CREATE INDEX log_events_agent_idx       ON log_events (agent) WHERE agent IS NOT NULL;
CREATE INDEX log_events_tags_idx        ON log_events USING gin (tags);
