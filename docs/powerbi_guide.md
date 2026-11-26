# Enterprise Power BI Integration Guide
## E-Commerce Intelligence Engine (E-CIE)

This guide provides instructions and architectural specifications for connecting **Power BI Desktop** directly to the E-CIE PostgreSQL warehouse (`ecie_project_db`) and constructing a **multi-page, interactive analytical report**.

---

## 1. Architectural Strategy & Design Rationale

### Why Direct PostgreSQL Views Instead of 3.1M Raw Rows?
Loading 3,100,000 raw text review rows directly into Power BI Desktop creates memory overhead and slow visual refresh rates. In enterprise analytics architectures, raw tables are pre-aggregated via database views:
1. **Push-down Aggregation**: PostgreSQL computes GROUP BY and aggregate functions in under 2 seconds.
2. **Instant Visual Interactivity**: Power BI loads compact analytical tables (~4,000 rows, ~2,100 rows, and 25,000 products) that slice, dice, and cross-filter instantaneously.
3. **Data Metric Realignment**: The dataset represents an **Amazon E-Commerce Customer Reviews & Product Catalog** rather than an internal ERP transactional order database. In place of transaction GMV/AOV, this report delivers:
   - **Temporal Rating & Sentiment Velocity** (Review trends over time)
   - **Category & Store Sentiment Distribution** (Merchant quality benchmarks)
   - **Price-to-Rating Relationships** (Price elasticity vs. customer satisfaction)

### Dual-Mode Flexibility (Direct Connection + Presentation Backup)
- **Primary Mode (Live)**: Direct connection to PostgreSQL on `localhost:5432`.
- **Decoupled Mode (Backup)**: Ready-to-import CSV files located in `data/powerbi/`. If PostgreSQL is offline or you are presenting on another machine, import the CSVs directly into Power BI.

---

## 2. Connecting Power BI to the Data Sources

### Option A: Direct PostgreSQL Connection (Recommended for Live Viva)
1. Open **Power BI Desktop**.
2. Click **Get Data** $\rightarrow$ **More...** $\rightarrow$ **Database** $\rightarrow$ **PostgreSQL database** $\rightarrow$ **Connect**.
3. In the PostgreSQL dialog box:
   - **Server**: `localhost:5432`
   - **Database**: `ecie_project_db`
   - **Data Connectivity mode**: Select **Import** (or **DirectQuery** if desired; Import is recommended for DAX measures).
4. If prompted for credentials:
   - Select **Database** tab on the left.
   - **User name**: `postgres`
   - **Password**: `tanvipostgres` (or your configured `POSTGRES_PASSWORD`).
   - Click **Connect**. (If an SSL warning dialog appears, click **OK** / **Continue**).
5. In the **Navigator** window, check the three analytical views:
   - `public.v_powerbi_category_store`
   - `public.v_powerbi_rating_trends`
   - `public.v_powerbi_price_vs_rating`
6. Click **Load**.

---

### Option B: Standalone CSV Import (Presentation Backup)
If presenting on a laptop without PostgreSQL running:
1. Open **Power BI Desktop**.
2. Click **Get Data** $\rightarrow$ **Text/CSV**.
3. Navigate to the project directory: `data/powerbi/` and import:
   - `category_store_sentiment.csv` (Renamed in Power BI to `Category_Store_Sentiment`)
   - `rating_trends_monthly.csv` (Renamed in Power BI to `Rating_Trends_Monthly`)
   - `price_vs_rating_catalog.csv` (Renamed in Power BI to `Price_Vs_Rating_Catalog`)
4. Click **Load**.

---

## 3. Essential DAX Measures

In Power BI Desktop, click **New Measure** in the Modeling tab to create these measures:

### Total Review Volume
```dax
Total Reviews = 
SUM('Category_Store_Sentiment'[review_count])
```

### Positive Sentiment Share (%)
```dax
Positive Sentiment % = 
DIVIDE(
    CALCULATE(
        SUM('Category_Store_Sentiment'[review_count]), 
        'Category_Store_Sentiment'[sentiment] = "positive"
    ),
    [Total Reviews],
    0
)
```
*(Format as Percentage, 1 decimal place: `0.0%`)*

### Negative Sentiment Share (%)
```dax
Negative Sentiment % = 
DIVIDE(
    CALCULATE(
        SUM('Category_Store_Sentiment'[review_count]), 
        'Category_Store_Sentiment'[sentiment] = "negative"
    ),
    [Total Reviews],
    0
)
```
*(Format as Percentage, 1 decimal place: `0.0%`)*

### Net Sentiment Score (NSS)
```dax
Net Sentiment Score = 
([Positive Sentiment %] - [Negative Sentiment %]) * 100
```
*(Industry benchmark score ranging from -100 to +100)*

### Average Rating
```dax
Average Rating = 
AVERAGE('Price_Vs_Rating_Catalog'[avg_rating])
```
*(Format as Decimal Number, 2 decimal places: `0.00`)*

### Verified Purchase Rate (%)
```dax
Verified Purchase % = 
DIVIDE(
    SUM('Category_Store_Sentiment'[verified_reviews]),
    [Total Reviews],
    0
)
```
*(Format as Percentage, 1 decimal place: `0.0%`)*

---

## 4. Multi-Page Report Layout & Visual Specifications

### 📄 Page 1: Category & Store Sentiment Intelligence
*Objective*: Compare consumer satisfaction across product departments and identify top/bottom performing merchant storefronts.

```
+----------------------------------------------------------------------------------------------------+
|  Page Title: Category & Store Sentiment Intelligence               [ Slicer: Category ] [ Store ]  |
+----------------------------------------------------------------------------------------------------+
|  [ Card ]                     [ Card ]                    [ Card ]               [ Card ]          |
|  Total Reviews: 3.1M          Net Sentiment: +78.4        Pos Sentiment: 89.2%   Verified: 96.1%   |
+-------------------------------------------------------------+--------------------------------------+
|  Visual 1: Category Sentiment Share                         |  Visual 2: Top 10 Merchant Stores    |
|  Type: 100% Stacked Bar Chart                               |  Type: Clustered Bar Chart           |
|  Y-Axis: category                                           |  Y-Axis: store_name                  |
|  X-Axis: review_count                                       |  X-Axis: review_count                |
|  Legend: sentiment (Green for positive, Red for negative)   |  Legend: sentiment                   |
|                                                             |  Top N Filter: Top 10 by review_count|
+-------------------------------------------------------------+--------------------------------------+
|  Visual 3: Store Benchmark Table                                                                   |
|  Columns: store_name | category | total_reviews | avg_rating | positive_pct | verified_reviews      |
+----------------------------------------------------------------------------------------------------+
```

- **Visual 1 Configuration (100% Stacked Bar Chart)**:
  - **Y-axis**: `'Category_Store_Sentiment'[category]`
  - **X-axis**: `'Category_Store_Sentiment'[review_count]`
  - **Legend**: `'Category_Store_Sentiment'[sentiment]`
  - **Data Colors**: Positive = `#10B981` (Emerald Green), Negative = `#EF4444` (Crimson Red).
- **Visual 2 Configuration (Clustered Bar Chart)**:
  - **Y-axis**: `'Category_Store_Sentiment'[store_name]`
  - **X-axis**: `'Category_Store_Sentiment'[review_count]`
  - **Legend**: `'Category_Store_Sentiment'[sentiment]`
  - **Filter**: Filter on `store_name` $\rightarrow$ Top N = 10 by `review_count`.
- **Visual 3 Configuration (Table / Matrix)**:
  - Add fields: `store_name`, `category`, `Total Reviews`, `Average Rating`, `Positive Sentiment %`.
  - Add conditional formatting data bars to `Positive Sentiment %`.

---

### 📄 Page 2: Temporal Rating Trends & Sentiment Velocity
*Objective*: Analyze longitudinal review trends from 2012 to 2023, observing seasonal shifts and sentiment volume growth.

```
+----------------------------------------------------------------------------------------------------+
|  Page Title: Temporal Rating Trends & Sentiment Velocity           [ Date Range Slider 2014-2023 ] |
+----------------------------------------------------------------------------------------------------+
|  Visual 1: Monthly Average Rating Evolution (Trend Line)                                           |
|  Type: Line Chart                                                                                  |
|  X-Axis: review_month_date (Continuous Date)                                                       |
|  Y-Axis: Average of avg_rating (Min: 3.5, Max: 5.0)                                                 |
|  Legend: category                                                                                  |
+----------------------------------------------------------------------------------------------------+
|  Visual 2: Sentiment Volume Over Time                       |  Visual 3: Yearly Review Growth      |
|  Type: Stacked Area Chart                                   |  Type: Clustered Column Chart        |
|  X-Axis: review_month_date                                  |  X-Axis: review_year                 |
|  Y-Axis: review_count                                       |  Y-Axis: review_count                |
|  Legend: sentiment                                          |  Legend: category                    |
+----------------------------------------------------------------------------------------------------+
```

- **Visual 1 Configuration (Line Chart)**:
  - **X-axis**: `'Rating_Trends_Monthly'[review_month_date]` (Set axis type to Continuous).
  - **Y-axis**: `'Rating_Trends_Monthly'[avg_rating]` (Summarization: Average).
  - **Legend**: `'Rating_Trends_Monthly'[category]`.
  - **Y-axis scale**: Fix Min to 3.0 or 3.5 to clearly display trends.
- **Visual 2 Configuration (Stacked Area Chart)**:
  - **X-axis**: `'Rating_Trends_Monthly'[review_month_date]`.
  - **Y-axis**: `'Rating_Trends_Monthly'[review_count]` (Sum).
  - **Legend**: `'Rating_Trends_Monthly'[sentiment]`.
- **Visual 3 Configuration (Clustered Column Chart)**:
  - **X-axis**: `'Rating_Trends_Monthly'[review_year]`.
  - **Y-axis**: `'Rating_Trends_Monthly'[review_count]` (Sum).
  - **Legend**: `'Rating_Trends_Monthly'[category]`.
  - **Filter**: `review_year` $\ge 2010$.

---

### 📄 Page 3: Price vs. Rating Relationships (Catalog Quality & Elasticity)
*Objective*: Examine whether higher product price points correlate with customer satisfaction, replacing transactional sales GMV with catalog quality analytics.

```
+----------------------------------------------------------------------------------------------------+
|  Page Title: Price vs. Rating Relationships & Catalog Elasticity   [ Slicer: Price Tier ] [ Cat ]  |
+----------------------------------------------------------------------------------------------------+
|  Visual 1: Catalog Price vs. Rating Scatter Plot                                                   |
|  Type: Scatter Chart                                                                               |
|  X-Axis: price ($)                                                                                 |
|  Y-Axis: avg_rating (1.0 to 5.0)                                                                   |
|  Size: total_reviews                                                                               |
|  Legend: category                                                                                  |
|  Tooltip: product_title, store_name, positive_pct                                                  |
+-------------------------------------------------------------+--------------------------------------+
|  Visual 2: Positive Sentiment Share by Price Tier           |  Visual 3: Top Rated Flagship Items  |
|  Type: Clustered Column Chart                               |  Type: Table                         |
|  X-Axis: price_tier (Budget, Mid-Range, Upper Mid, Premium) |  Columns: product_title, price,      |
|  Y-Axis: Average of positive_pct                            |           avg_rating, total_reviews, |
|  Data Labels: On                                            |           store_name                 |
+-------------------------------------------------------------+--------------------------------------+
```

- **Visual 1 Configuration (Scatter Chart)**:
  - **X-axis**: `'Price_Vs_Rating_Catalog'[price]`.
  - **Y-axis**: `'Price_Vs_Rating_Catalog'[avg_rating]`.
  - **Size**: `'Price_Vs_Rating_Catalog'[total_reviews]`.
  - **Legend**: `'Price_Vs_Rating_Catalog'[category]`.
  - **Tooltips**: Add `product_title`, `store_name`, `positive_pct`.
- **Visual 2 Configuration (Clustered Column Chart)**:
  - **X-axis**: `'Price_Vs_Rating_Catalog'[price_tier]`.
  - **Y-axis**: `'Price_Vs_Rating_Catalog'[positive_pct]` (Summarization: Average).
  - **Data Labels**: Enable to display exact percentage per tier (e.g. Budget: 87.4%, Premium: 91.8%).
- **Visual 3 Configuration (Table)**:
  - Add fields: `product_title`, `store_name`, `price`, `avg_rating`, `total_reviews`, `positive_pct`.
  - Sort by: `avg_rating` descending or `total_reviews` descending.

---

## 5. Viva Presentation Defense & Examiner Q&A

### Q1: "Why did you build views instead of importing the 3.1M rows directly into Power BI?"
> **Answer**: *"Connecting Power BI to 3.1 million raw text records would create substantial memory bloat and degrade interactive slicing. In an enterprise data architecture, analytical views push heavy computations down to the database engine. Our PostgreSQL views pre-aggregate millions of rows into high-performance analytical summaries in 1.2 seconds, giving Power BI instantaneous cross-filtering while keeping exact aggregate numbers."*

### Q2: "Why doesn't this dashboard show total sales revenue (GMV) or Average Order Value (AOV)?"
> **Answer**: *"The dataset is Amazon's real-world customer review and catalog metadata corpus, not an internal cart checkout/ERP transaction log. Rather than hallucinating fake sales volume numbers, we performed Catalog Price Elasticity and Sentiment Analysis—evaluating how price tiers relate to customer satisfaction, verified purchase trust ratios, and store-level sentiment distribution."*

### Q3: "How does Power BI connect with your Machine Learning Pipeline?"
> **Answer**: *"The sentiment column analyzed in the database views directly maps to our NLP sentiment classification engine. The SGDClassifier (98.31% holdout accuracy) classifies reviews into positive and negative polarities, which are harmonized in the PostgreSQL warehouse and piped straight into Power BI's DAX measures and interactive visualizations."*

---

## 6. Dataset Refresh Command
To refresh or re-export the CSV files at any time, run:
```bash
.venv/bin/python scripts/export_powerbi.py
```
This queries the PostgreSQL views and refreshes `data/powerbi/*.csv` automatically.
