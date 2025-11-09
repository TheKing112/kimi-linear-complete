-- =============================================================================
-- GitHub Integration Tables
-- =============================================================================

CREATE TABLE IF NOT EXISTS github_repositories (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    repo_url TEXT NOT NULL,
    repo_name VARCHAR(500),
    cloned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_sync TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'active', -- active, archived, error
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_github_repos_user ON github_repositories(user_id);
CREATE INDEX idx_github_repos_url ON github_repositories(repo_url);

CREATE TABLE IF NOT EXISTS github_webhooks (
    id SERIAL PRIMARY KEY,
    repo_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    payload JSONB NOT NULL,
    processed BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_github_webhooks_repo ON github_webhooks(repo_id);
CREATE INDEX idx_github_webhooks_processed ON github_webhooks(processed) WHERE processed = false;