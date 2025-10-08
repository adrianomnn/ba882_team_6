# YouTube Data Pipeline Project

## Overview
This project is part of the **BA 882: Deploying Analytics Pipelines** class at Boston University.  
It builds an automated data pipeline that extracts data from the **YouTube Data API**,  
transforms it, and loads it into a cloud data warehouse for analytics, ML, and GenAI use cases.

## Project Phases
1. **Data Engineering & Reporting** – Build a scheduled ETL pipeline to ingest YouTube data.
2. **MLOps** – Train and deploy models predicting engagement metrics.
3. **GenAI Workflows** – Integrate AI-based data augmentation or insights generation.

## Architecture
- **Data Source:** YouTube Data API v3  
- **Storage:** Cloud Data Warehouse (e.g. BigQuery)  
- **Orchestration:** Airflow / Prefect  
- **Visualization:** ? (Streamlit / Metabase / Superset) 

## Data Model

This project uses the YouTube Data API v3 to extract, transform, and store structured information from YouTube channels, videos, comments, and categories.
Each table is updated automatically by the pipeline on a daily schedule.

| Table Name           | Description                                                                                       | Primary Key   | Update Frequency                            | Example Fields                                                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------------- | ------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **channels**         | Contains general information about YouTube channels (creator metadata, audience stats, etc.)      | `channel_id`  | Daily                                      | `channel_id`, `channel_title`, `description`, `published_at`, `country`, `view_count`, `subscriber_count`, `video_count` |
| **videos**           | Metadata for each video uploaded by a channel, such as title, description, duration, and category | `video_id`    | Daily                                      | `video_id`, `channel_id`, `title`, `description`, `published_at`, `category_id`, `duration`, `tags`                      |
| **video_statistics** | Engagement metrics for each video, updated regularly to reflect performance                       | `video_id`    | Daily                                      | `video_id`, `view_count`, `like_count`, `favorite_count`, `comment_count`                                                |
| **comments**         | Captures viewer feedback and engagement text from each video                                      | `comment_id`  | Daily                                      | `comment_id`, `video_id`, `author_display_name`, `text_display`, `like_count`, `published_at`                            |
| **video_categories** | Contains the list of available video categories for the region, as defined by YouTube             | `category_id` | Static (updated if API changes) | `category_id`, `title`, `assignable`                                                                                     |

