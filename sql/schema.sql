-- =============================================================================
-- ECIE Database Schema & Views
-- Database: ecie_project_db
-- Dataset: Amazon Reviews 2023 Relational Warehouse
-- =============================================================================

-- 1. Stores Table (133k rows)
CREATE TABLE IF NOT EXISTS stores (
    store_id BIGINT PRIMARY KEY,
    store_name TEXT
);

-- 2. Users Table (869k rows)
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(100) PRIMARY KEY,
    total_reviews INT,
    avg_rating_given NUMERIC(4, 2),
    total_helpful_votes INT,
    first_review_timestamp BIGINT,
    last_review_timestamp BIGINT
);

-- 3. Products Table (857k rows)
CREATE TABLE IF NOT EXISTS products (
    parent_asin VARCHAR(50) PRIMARY KEY,
    product_title TEXT,
    category VARCHAR(255),
    main_category VARCHAR(255),
    price NUMERIC(10, 2),
    store_id BIGINT
);

-- 4. Reviews Metadata Table (3.1M rows)
CREATE TABLE IF NOT EXISTS reviews (
    review_id BIGINT PRIMARY KEY,
    parent_asin VARCHAR(50),
    user_id VARCHAR(100),
    rating NUMERIC(3, 1),
    sentiment VARCHAR(20),
    timestamp BIGINT,
    helpful_vote INT,
    verified_purchase BOOLEAN,
    category VARCHAR(255)
);

-- 5. Review Text Content Table (3.1M rows)
CREATE TABLE IF NOT EXISTS review_text (
    review_id BIGINT PRIMARY KEY,
    review_title TEXT,
    review_text TEXT
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);
CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment);
CREATE INDEX IF NOT EXISTS idx_reviews_parent_asin ON reviews(parent_asin);
CREATE INDEX IF NOT EXISTS idx_reviews_user_id ON reviews(user_id);

-- =============================================================================
-- Analytical View: v_cleaned_reviews
-- Multi-class sentiment mapping:
--   1.0 - 2.0 -> 'negative'
--   3.0       -> 'neutral'
--   4.0 - 5.0 -> 'positive'
-- =============================================================================
CREATE OR REPLACE VIEW v_cleaned_reviews AS
SELECT 
    r.review_id,
    r.parent_asin,
    r.user_id,
    r.rating,
    r.verified_purchase,
    r.category,
    r.timestamp,
    r.helpful_vote,
    rt.review_title,
    rt.review_text,
    CASE 
        WHEN r.rating >= 4.0 THEN 'positive'
        WHEN r.rating <= 2.0 THEN 'negative'
        ELSE 'neutral'
    END AS sentiment
FROM reviews r
JOIN review_text rt ON r.review_id = rt.review_id;
