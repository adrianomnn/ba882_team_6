"""
Create Golden Layer tables for YouTube data analytics and ML feature engineering.
"""

import functions_framework
from google.cloud import bigquery
from flask import jsonify

project_id = "adrineto-qst882-fall25"
dataset_id = "youtube_golden"
location = "us-central1"

@functions_framework.http
def task(request):
    client = bigquery.Client(project=project_id, location=location)

    # --- Step 1: Define Golden Layer Table Schemas ---
    table_schemas = {
        "video_engagement_features": [
            bigquery.SchemaField("video_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("total_views", "INT64"),
            bigquery.SchemaField("total_likes", "INT64"),
            bigquery.SchemaField("total_comments", "INT64"),
            bigquery.SchemaField("like_view_ratio", "FLOAT64"),
            bigquery.SchemaField("comment_view_ratio", "FLOAT64"),
            bigquery.SchemaField("last_updated", "TIMESTAMP")
        ],
        "video_comment_sentiment_features": [
            bigquery.SchemaField("video_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("total_comments", "INT64"),
            bigquery.SchemaField("avg_sentiment", "FLOAT64"),
            bigquery.SchemaField("sentiment_stddev", "FLOAT64"),
            bigquery.SchemaField("pct_positive_comments", "FLOAT64"),
            bigquery.SchemaField("last_updated", "TIMESTAMP")
        ],
        "video_metadata_features": [
            bigquery.SchemaField("video_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("channel_id", "STRING"),
            bigquery.SchemaField("title_length", "INT64"),
            bigquery.SchemaField("duration_seconds", "INT64"),
            bigquery.SchemaField("days_since_published", "INT64"),
            bigquery.SchemaField("is_viral", "BOOL"),
            bigquery.SchemaField("last_updated", "TIMESTAMP")
        ]
    }

    # --- Step 2: Ensure Tables Exist ---
    for table_name, schema in table_schemas.items():
        table_id = f"{project_id}.{dataset_id}.{table_name}"
        try:
            client.get_table(table_id)
            print(f"Table already exists: {table_id}")
        except Exception:
            table = bigquery.Table(table_id, schema=schema)
            client.create_table(table)
            print(f"Created table: {table_id}")

    # --- Step 3: Transformation SQLs for Golden Layer ---

    queries = [

        # VIDEO ENGAGEMENT FEATURES
        """
        CREATE OR REPLACE TABLE `adrineto-qst882-fall25.youtube_golden.video_engagement_features` AS
        SELECT
            v.video_id,
            MAX(s.view_count) AS total_views,
            MAX(s.like_count) AS total_likes,
            MAX(s.comment_count) AS total_comments,
            SAFE_DIVIDE(MAX(s.like_count), MAX(s.view_count)) AS like_view_ratio,
            SAFE_DIVIDE(MAX(s.comment_count), MAX(s.view_count)) AS comment_view_ratio,
            CURRENT_TIMESTAMP() AS last_updated
        FROM `adrineto-qst882-fall25.youtube_staging.fact_video_statistics` s
        JOIN `adrineto-qst882-fall25.youtube_staging.dim_videos` v
          ON s.video_id = v.video_id
        GROUP BY v.video_id;
        """,

        # VIDEO COMMENT SENTIMENT FEATURES
        """
        CREATE OR REPLACE TABLE `adrineto-qst882-fall25.youtube_golden.video_comment_sentiment_features` AS
        SELECT
            video_id,
            COUNT(*) AS total_comments,
            0.5 AS avg_sentiment,
            0.5 AS sentiment_stddev,
            SUM(CASE WHEN sentiment_score > 0 THEN 1 ELSE 0 END) / COUNT(*) AS pct_positive_comments,
            CURRENT_TIMESTAMP() AS last_updated
        FROM `adrineto-qst882-fall25.youtube_staging.dim_comments`
        WHERE sentiment_score IS NOT NULL
        GROUP BY video_id;
        """,

        # VIDEO METADATA FEATURES
        """
        CREATE OR REPLACE TABLE `adrineto-qst882-fall25.youtube_golden.video_metadata_features` AS
        SELECT
            v.video_id,
            v.channel_id,
            LENGTH(v.title) AS title_length,
            
            SAFE_CAST(
                SPLIT(vs.duration, ":")[OFFSET(0)] AS INT64
            ) * 60 +
            SAFE_CAST(SPLIT(vs.duration, ":")[SAFE_OFFSET(1)] AS INT64) AS duration_seconds,
            
            DATE_DIFF(CURRENT_DATE(), DATE(v.published_at), DAY) AS days_since_published,

            -- rule-based viral flag: >100k views within 7 days
            CASE
                WHEN MAX(s.view_count) > 100000 AND DATE_DIFF(CURRENT_DATE(), DATE(v.published_at), DAY) <= 7 THEN TRUE
                ELSE FALSE
            END AS is_viral,
            
            CURRENT_TIMESTAMP() AS last_updated
        FROM `adrineto-qst882-fall25.youtube_staging.dim_videos` v
        JOIN `adrineto-qst882-fall25.youtube_staging.fact_video_statistics` s
          ON v.video_id = s.video_id
        LEFT JOIN `adrineto-qst882-fall25.youtube_staging.fact_video_statistics` vs
          ON v.video_id = vs.video_id
        GROUP BY v.video_id, v.channel_id, v.title, v.published_at, vs.duration;
        """
    ]

    # --- Step 4: Execute Queries ---
    results = []
    for i, query in enumerate(queries):
        job = client.query(query)
        job.result()
        results.append(f"Golden Layer Query {i+1} executed successfully")

    return jsonify({
        "status": "success",
        "message": "Golden Layer tables created and updated successfully.",
        "results": results
    })
