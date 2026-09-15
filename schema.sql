-- ============================================================
-- Social Engine — Round 1, Phase 2: Schema Design
-- ============================================================
-- Design rationale (see Logic Explanation section of the report
-- for full reasoning):
--   - users / posts are the two core entities from Phase 1.
--   - hashtags and mentions were multi-valued attributes on each
--     post (a post can have 0..N of each). Storing them as
--     comma-joined strings inside the posts table would violate
--     1NF (repeating groups) and make querying by hashtag
--     painfully slow (LIKE '%tag%' scans). They are normalized
--     into their own junction tables instead, giving a clean
--     many-to-many relationship and enabling indexed, set-based
--     queries (e.g. "posts per hashtag", "co-occurring hashtags").
--   - Foreign key posts.user_id -> users.user_id enforces
--     referential integrity between the two Phase 1 datasets.

CREATE TABLE users (
    user_id           VARCHAR PRIMARY KEY,
    location          VARCHAR,
    language          VARCHAR,
    account_created   DATE,
    follower_count    INTEGER
);

CREATE TABLE posts (
    post_id           VARCHAR PRIMARY KEY,
    user_id           VARCHAR REFERENCES users(user_id),
    platform          VARCHAR,
    text_content      VARCHAR,
    has_text          BOOLEAN,
    post_timestamp    TIMESTAMP,
    likes             INTEGER,
    shares            INTEGER,
    comments          INTEGER
);

-- Junction table: one row per (post, hashtag) pair
CREATE TABLE post_hashtags (
    post_id           VARCHAR REFERENCES posts(post_id),
    hashtag           VARCHAR
);

-- Junction table: one row per (post, mentioned user) pair
CREATE TABLE post_mentions (
    post_id           VARCHAR REFERENCES posts(post_id),
    mentioned_handle  VARCHAR
);

CREATE INDEX idx_posts_user ON posts(user_id);
CREATE INDEX idx_posts_platform ON posts(platform);
CREATE INDEX idx_hashtags_tag ON post_hashtags(hashtag);
