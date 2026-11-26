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

-- =============================================================================
-- Analytical View: v_powerbi_category_store
-- Pre-aggregated review counts and sentiment metrics grouped by category and store.
-- Designed for direct high-speed Power BI consumption (4,000 rows vs 3.1M raw).
-- =============================================================================
CREATE OR REPLACE VIEW v_powerbi_category_store AS
SELECT 
    r.category,
    s.store_id,
    s.store_name,
    r.sentiment,
    COUNT(*) AS review_count,
    ROUND(AVG(r.rating), 2) AS avg_rating,
    COUNT(CASE WHEN r.verified_purchase = true THEN 1 END) AS verified_reviews,
    ROUND(AVG(p.price), 2) AS avg_product_price
FROM reviews r
JOIN products p ON r.parent_asin = p.parent_asin
JOIN stores s ON p.store_id = s.store_id
GROUP BY r.category, s.store_id, s.store_name, r.sentiment;

-- =============================================================================
-- Analytical View: v_powerbi_rating_trends
-- Monthly temporal aggregation of review volume, ratings, and sentiment distribution.
-- Enables interactive Power BI timeline slicers, trend lines, and velocity charts.
-- =============================================================================
CREATE OR REPLACE VIEW v_powerbi_rating_trends AS
SELECT 
    DATE_TRUNC('month', TO_TIMESTAMP(r.timestamp::numeric / 1000.0))::date AS review_month_date,
    EXTRACT(YEAR FROM TO_TIMESTAMP(r.timestamp::numeric / 1000.0))::integer AS review_year,
    EXTRACT(MONTH FROM TO_TIMESTAMP(r.timestamp::numeric / 1000.0))::integer AS review_month,
    TO_CHAR(TO_TIMESTAMP(r.timestamp::numeric / 1000.0), 'Mon YYYY') AS month_year_label,
    r.category,
    r.sentiment,
    COUNT(*) AS review_count,
    ROUND(AVG(r.rating), 2) AS avg_rating,
    COUNT(CASE WHEN r.verified_purchase = true THEN 1 END) AS verified_reviews
FROM reviews r
GROUP BY 
    DATE_TRUNC('month', TO_TIMESTAMP(r.timestamp::numeric / 1000.0))::date,
    EXTRACT(YEAR FROM TO_TIMESTAMP(r.timestamp::numeric / 1000.0))::integer,
    EXTRACT(MONTH FROM TO_TIMESTAMP(r.timestamp::numeric / 1000.0))::integer,
    TO_CHAR(TO_TIMESTAMP(r.timestamp::numeric / 1000.0), 'Mon YYYY'),
    r.category,
    r.sentiment;

-- =============================================================================
-- Analytical View: v_powerbi_price_vs_rating
-- Product-level price elasticity, rating tiers, and sentiment breakdown.
-- Substitutes order-level GMV/AOV metrics with catalog price vs. satisfaction relationships.
-- =============================================================================
CREATE OR REPLACE VIEW v_powerbi_price_vs_rating AS
SELECT 
    p.parent_asin,
    p.product_title,
    p.category,
    s.store_name,
    p.price,
    CASE 
        WHEN p.price < 25.00 THEN '1. Budget (<$25)'
        WHEN p.price < 75.00 THEN '2. Mid-Range ($25-$75)'
        WHEN p.price < 150.00 THEN '3. Upper Mid ($75-$150)'
        ELSE '4. Premium ($150+)'
    END AS price_tier,
    COUNT(r.review_id) AS total_reviews,
    ROUND(AVG(r.rating), 2) AS avg_rating,
    COUNT(CASE WHEN r.sentiment = 'positive' THEN 1 END) AS positive_reviews,
    COUNT(CASE WHEN r.sentiment = 'negative' THEN 1 END) AS negative_reviews,
    ROUND(COUNT(CASE WHEN r.sentiment = 'positive' THEN 1 END)::numeric / NULLIF(COUNT(r.review_id), 0) * 100, 1) AS positive_pct
FROM products p
JOIN stores s ON p.store_id = s.store_id
JOIN reviews r ON p.parent_asin = r.parent_asin
GROUP BY p.parent_asin, p.product_title, p.category, s.store_name, p.price;

