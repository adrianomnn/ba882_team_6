# YouTube Data Pipeline Project

## Overview
This project is part of the **BA 882: Deploying Analytics Pipelines** class at **Boston University**.  
It builds a **fully automated ELT pipeline** that extracts structured data from the **YouTube Data API v3**, transforms it using **Google Cloud Functions**, and orchestrates daily executions with **Airflow (Astronomer)**.  
The pipeline is deployed and executed in **Google Cloud Platform (GCP)** using **BigQuery** as the data warehouse.

---

## Architecture Overview

**Components:**
| Layer | Technology | Description |
|-------|-------------|-------------|
| **Extraction** | YouTube Data API v3 | Collects channel, video, comment, and statistics data. |
| **Transformation** | Google Cloud Functions | Performs incremental transformations and merges data into fact/dimension tables. |
| **Storage** | BigQuery | Stores both raw and staging (curated) datasets. |
| **Orchestration** | Apache Airflow (Astronomer) | Schedules daily pipeline runs and manages task dependencies. |
| **Visualization / Analytics** | BigQuery + BI Tools | Enables downstream analytics, ML models, and GenAI experiments. |

**Data Flow:**
1. **Raw ingestion** → API extracts JSON data and stores it in `youtube_raw` tables.  
2. **Transform (Cloud Function)** → SQL-based transformation merges raw data into dimension and fact tables in `youtube_staging`.  
3. **Airflow DAG** → Triggers extraction and transformation tasks daily at midnight.  
4. **BigQuery** → Serves as the single source of truth for reporting and analytics.

---

## Datasets and Schema Design

The project follows a **data warehouse star schema** model in BigQuery, separating **raw**, **staging**, and **analytics-ready** layers:

### **1. Raw Layer (`youtube_raw`)**
Stores unprocessed API responses for reproducibility and auditing.

| Table | Description |
|--------|-------------|
| `videos` | Raw video metadata (title, description, published date, etc.) |
| `channels` | Raw channel metadata from creators. |
| `video_statistics` | Engagement metrics for each video (views, likes, comments). |
| `comments` | Viewer comments and author metadata. |
| `video_categories` | Reference table for available video categories by region. |

---

### **2. Staging Layer (`youtube_staging`)**
Applies incremental **MERGE transformations** to maintain historical data consistency while preventing duplicates.

#### **Dimension Tables**

| Table Name | Description | Primary Key | Key Columns | Example Fields |
|-------------|-------------|--------------|--------------|----------------|
| **dim_channels** | Stores metadata about YouTube channels. | `channel_id` | `channel_id` | `channel_id`, `channel_title`, `channel_description`, `last_updated` |
| **dim_videos** | Metadata about individual videos uploaded to channels. | `video_id` | `video_id`, `channel_id` | `video_id`, `title`, `description`, `channel_id`, `published_at`, `last_updated` |
| **dim_comments** | Details about individual comments and their authors. | `comment_id` | `comment_id` | `comment_id`, `author_display_name`, `comment_text`, `last_updated` |

#### **Fact Tables**

| Table Name | Description | Primary Key | Foreign Keys | Example Fields |
|-------------|-------------|--------------|--------------|----------------|
| **fact_video_statistics** | Tracks daily engagement metrics (views, likes, comments) for each video. | (`video_id`, `date`) | `video_id`, `channel_id` | `video_id`, `channel_id`, `date`, `duration`, `view_count`, `like_count`, `comment_count` |
| **fact_comments** | Records each comment and engagement data related to videos. | `comment_id` | `video_id` | `comment_id`, `video_id`, `like_count`, `published_at` |

---

### **3. Incremental Merge Logic**

Each transformation uses **BigQuery MERGE statements** with deduplication rules:
- **Dimension tables:** Merge by ID (e.g., `video_id`, `channel_id`, `comment_id`)  
- **Fact tables:** Merge by composite key (`video_id`, `date`)  
- **Durations:** Converted from ISO 8601 format (e.g., `PT1M33S`) → human-readable format (`1:33`).  
  - Handles edge cases such as `PT1M` → `1:00` and `PT2H` → `2:00:00`.

## Deployment

1. Google Cloud Functions

- Function: raw-schema, raw-extract, raw-parse, and raw-transform

- Trigger: HTTP (called from Airflow DAG)

- Role: Create schema in BigQuery, extract data from YouTube API and load in our Cloud Storage Bucket, load the data in BigQuery Data Warehouse, and executes BigQuery transformations using incremental MERGE statements.

2. Airflow (Astronomer)

DAG orchestrates:

- API Extraction (raw data ingestion)

- Cloud Function trigger (transformation)

- Data validation and logging

- Schedule: Daily at 00:00 (midnight)

- Logs: Stored in Airflow and GCP Logs Explorer
