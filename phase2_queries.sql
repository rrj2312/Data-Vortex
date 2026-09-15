-- ============================================================
-- Data Vortex — Round 1, Phase 2: Selected Queries
-- Questions: E3 (Easy), M1 (Medium), H5 (Hard)
-- Tool: DuckDB
-- ============================================================


-- ============================================================
-- E3 — Average Engagement by Platform
-- ============================================================
SELECT platform,
       ROUND(AVG(likes), 1)   AS avg_likes,
       ROUND(AVG(shares), 1)  AS avg_shares,
       ROUND(AVG(comments), 1) AS avg_comments,
       ROUND(AVG(likes) + AVG(shares) + AVG(comments), 1) AS avg_total_engagement
FROM posts
GROUP BY platform
ORDER BY avg_total_engagement DESC;


-- ============================================================
-- M1 — Which Locations Generate the Most Engagement?
-- ============================================================
SELECT u.location,
       COUNT(p.post_id) AS num_posts,
       SUM(p.likes + p.shares + p.comments) AS total_engagement,
       RANK() OVER (ORDER BY SUM(p.likes + p.shares + p.comments) DESC) AS engagement_rank
FROM posts p
JOIN users u ON p.user_id = u.user_id
GROUP BY u.location
ORDER BY total_engagement DESC;


-- ============================================================
-- H5 — Identify Data Anomalies
-- Run against the RAW pre-cleaning dataset (posts_raw), since
-- this question specifically targets Phase 1's corruption.
-- ============================================================

-- Row-level: post_id + which anomaly type(s) apply
SELECT
    post_id,
    CONCAT_WS(', ',
        CASE WHEN TRY_CAST(likes AS DOUBLE) < 0 THEN 'Negative Likes' END,
        CASE WHEN platform IS NULL OR TRIM(platform) IN ('', 'NULL')
             THEN 'Missing Platform' END,
        CASE WHEN text_content IS NULL OR TRIM(text_content) IN ('', 'NULL')
             THEN 'Missing Text' END,
        CASE WHEN text_content LIKE '%&amp;%' OR text_content LIKE '%<div>%'
                  OR text_content LIKE '%<br>%'
             THEN 'HTML Entity/Tag Corruption' END
    ) AS anomaly_type
FROM posts_raw
WHERE TRY_CAST(likes AS DOUBLE) < 0
   OR platform IS NULL OR TRIM(platform) IN ('', 'NULL')
   OR text_content IS NULL OR TRIM(text_content) IN ('', 'NULL')
   OR text_content LIKE '%&amp;%' OR text_content LIKE '%<div>%' OR text_content LIKE '%<br>%'
ORDER BY post_id;

-- Supplementary: count of posts affected by each anomaly type
SELECT
    SUM(CASE WHEN TRY_CAST(likes AS DOUBLE) < 0 THEN 1 ELSE 0 END) AS negative_likes,
    SUM(CASE WHEN platform IS NULL OR TRIM(platform) IN ('', 'NULL') THEN 1 ELSE 0 END) AS missing_platform,
    SUM(CASE WHEN text_content IS NULL OR TRIM(text_content) IN ('', 'NULL') THEN 1 ELSE 0 END) AS missing_text,
    SUM(CASE WHEN text_content LIKE '%&amp;%' OR text_content LIKE '%<div>%' OR text_content LIKE '%<br>%'
             THEN 1 ELSE 0 END) AS html_corruption,
    COUNT(*) AS total_rows_scanned
FROM posts_raw;
