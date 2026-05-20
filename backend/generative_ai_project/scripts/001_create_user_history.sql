CREATE TABLE IF NOT EXISTS user_history (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NULL,
    session_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    detected_intent TEXT NULL,
    service_type TEXT NULL,
    location TEXT NULL,
    requested_time TEXT NULL,
    selected_provider_id INTEGER NULL,
    selected_provider_name TEXT NULL,
    booking_status TEXT NULL,
    reasoning JSONB NULL,
    workflow_trace JSONB NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_history_session ON user_history(session_id);
CREATE INDEX IF NOT EXISTS idx_user_history_created_at ON user_history(created_at);
