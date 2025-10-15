"""
Transform raw data into staging tables for YouTube data with incremental merge logic.
"""
import functions_framework
from google.cloud import bigquery
from flask import jsonify

project_id = 'adrineto-qst882-fall25'
dataset_id = 'youtube_staging'
location = 'us-central1'

@functions_framework.http
def task(request):
    client = bigquery.Client(project=project_id, location=location)

    # Define staging table schemas
    table_schemas = {
        "dim_videos": [
            bigquery.SchemaField("video_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("description", "STRING"),
            bigquery.SchemaField("channel_id", "STRING"),
            bigquery.SchemaField("published_at", "TIMESTAMP"),
            bigquery.SchemaField("last_updated", "TIMESTAMP")
        ],
        "dim_channels": [
            bigquery.SchemaField("channel_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("channel_title", "STRING"),
            bigquery.SchemaField("channel_description", "STRING"),
            bigquery.SchemaField("last_updated", "TIMESTAMP")
        ],
        "dim_comments": [
            bigquery.SchemaField("comment_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("author_display_name", "STRING"),
            bigquery.SchemaField("comment_text", "STRING"),
            bigquery.SchemaField("last_updated", "TIMESTAMP")
        ],
        "fact_video_statistics": [
            bigquery.SchemaField("video_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("channel_id", "STRING"),
            bigquery.SchemaField("duration", "STRING"),
            bigquery.SchemaField("date", "DATE"),
            bigquery.SchemaField("view_count", "INTEGER"),
            bigquery.SchemaField("like_count", "INTEGER"),
            bigquery.SchemaField("comment_count", "INTEGER")
        ],
        "fact_comments": [
            bigquery.SchemaField("comment_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("video_id", "STRING"),
            bigquery.SchemaField("like_count", "INTEGER"),
            bigquery.SchemaField("published_at", "TIMESTAMP")
        ],
    }

    # --- Step 1: Ensure all tables exist ---
    for table_name, schema in table_schemas.items():
        table_id = f"{project_id}.{dataset_id}.{table_name}"
        try:
            client.get_table(table_id)
            print(f"Table already exists: {table_id}")
        except Exception:
            table = bigquery.Table(table_id, schema=schema)
            client.create_table(table)
            print(f"Created table: {table_id}")

    # --- Step 2: Run transformations ---
    queries = [

    # DIM_VIDEOS
    """
    MERGE `adrineto-qst882-fall25.youtube_staging.dim_videos` AS T
    USING (
      SELECT DISTINCT
        video_id,
        title,
        description,
        channel_id,
        published_at
      FROM `adrineto-qst882-fall25.youtube_raw.videos`
    ) AS S
    ON T.video_id = S.video_id
    WHEN MATCHED THEN
      UPDATE SET
        T.title = S.title,
        T.description = S.description,
        T.channel_id = S.channel_id,
        T.published_at = S.published_at,
        T.last_updated = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN
      INSERT (video_id, title, description, channel_id, published_at, last_updated)
      VALUES (S.video_id, S.title, S.description, S.channel_id, S.published_at, CURRENT_TIMESTAMP());
    """,

    # DIM_CHANNELS
    """
    MERGE `adrineto-qst882-fall25.youtube_staging.dim_channels` AS T
    USING (
      SELECT DISTINCT
        channel_id,
        channel_title,
        channel_description
      FROM `adrineto-qst882-fall25.youtube_raw.channels`
    ) AS S
    ON T.channel_id = S.channel_id
    WHEN MATCHED THEN
      UPDATE SET
        T.channel_title = S.channel_title,
        T.channel_description = S.channel_description,
        T.last_updated = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN
      INSERT (channel_id, channel_title, channel_description, last_updated)
      VALUES (S.channel_id, S.channel_title, S.channel_description, CURRENT_TIMESTAMP());
    """,

    # DIM_COMMENTS
    """
    MERGE `adrineto-qst882-fall25.youtube_staging.dim_comments` AS T
    USING (
      SELECT DISTINCT
        comment_id,
        author_display_name,
        text_display AS comment_text
      FROM `adrineto-qst882-fall25.youtube_raw.comments`
    ) AS S
    ON T.comment_id = S.comment_id
    WHEN MATCHED THEN
      UPDATE SET
        T.author_display_name = S.author_display_name,
        T.comment_text = S.comment_text,
        T.last_updated = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN
      INSERT (comment_id, author_display_name, comment_text, last_updated)
      VALUES (S.comment_id, S.author_display_name, S.comment_text, CURRENT_TIMESTAMP());
    """,

    # FACT_VIDEO_STATISTICS
    """
    MERGE `adrineto-qst882-fall25.youtube_staging.fact_video_statistics` AS T
    USING (
      SELECT
        v.video_id,
        v.channel_id,
        CASE
          WHEN REGEXP_CONTAINS(duration, r'PT\d+H\d+M\d+S') THEN
            CONCAT(
              CAST(REGEXP_EXTRACT(duration, r'PT(\d+)H') AS STRING), ':',
              LPAD(CAST(REGEXP_EXTRACT(duration, r'(\d+)M') AS STRING), 2, '0'), ':',
              LPAD(CAST(REGEXP_EXTRACT(duration, r'(\d+)S') AS STRING), 2, '0')
            )
          WHEN REGEXP_CONTAINS(duration, r'PT\d+M\d+S') THEN
            CONCAT(
              CAST(REGEXP_EXTRACT(duration, r'PT(\d+)M') AS STRING), ':',
              LPAD(CAST(REGEXP_EXTRACT(duration, r'(\d+)S') AS STRING), 2, '0')
            )
          WHEN REGEXP_CONTAINS(duration, r'PT\d+S') THEN
            CONCAT('0:', LPAD(CAST(REGEXP_EXTRACT(duration, r'PT(\d+)S') AS STRING), 2, '0'))
          ELSE
            NULL
        END AS duration,
        CURRENT_DATE() AS date,
        s.view_count,
        s.like_count,
        s.comment_count
      FROM `adrineto-qst882-fall25.youtube_raw.video_statistics` s
      JOIN `adrineto-qst882-fall25.youtube_raw.videos` v
        ON s.video_id = v.video_id
    ) AS S
    ON T.video_id = S.video_id AND T.date = S.date
    WHEN MATCHED THEN
      UPDATE SET
        T.view_count = S.view_count,
        T.like_count = S.like_count,
        T.comment_count = S.comment_count,
        T.duration = S.duration
    WHEN NOT MATCHED THEN
      INSERT (video_id, channel_id, duration, date, view_count, like_count, comment_count)
      VALUES (S.video_id, S.channel_id, S.duration, S.date, S.view_count, S.like_count, S.comment_count);
    """,

    # FACT_COMMENTS
    """
    MERGE `adrineto-qst882-fall25.youtube_staging.fact_comments` AS T
    USING (
      SELECT
        comment_id,
        video_id,
        like_count,
        published_at
      FROM `adrineto-qst882-fall25.youtube_raw.comments`
    ) AS S
    ON T.comment_id = S.comment_id
    WHEN MATCHED THEN
      UPDATE SET
        T.video_id = S.video_id,
        T.like_count = S.like_count,
        T.published_at = S.published_at
    WHEN NOT MATCHED THEN
      INSERT (comment_id, video_id, like_count, published_at)
      VALUES (S.comment_id, S.video_id, S.like_count, S.published_at);
    """
    ]

    results = []
    for i, query in enumerate(queries):
        job = client.query(query)
        job.result()
        results.append(f"Query {i+1} executed successfully")

    return jsonify({
        "status": "success",
        "message": "Incremental transformations completed successfully",
        "results": results
    })