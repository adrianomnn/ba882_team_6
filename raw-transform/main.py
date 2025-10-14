"""
Create BigQuery staging tables for YouTube data
"""
import functions_framework
from google.cloud import bigquery
from flask import jsonify

project_id = 'adrineto-qst882-fall25'
dataset_id = 'youtube_raw'

@functions_framework.http
def task(request):
    """
    Transform raw data into staging tables using incremental MERGE logic.
    """
    client = bigquery.Client(project=project_id)

    queries = [
        # dim_videos
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
        # dim_channels
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
        # dim_authors
        """
        MERGE `adrineto-qst882-fall25.youtube_staging.dim_authors` AS T
        USING (
          SELECT DISTINCT
            author_id,
            author_name
          FROM `adrineto-qst882-fall25.youtube_raw.comment_authors`
        ) AS S
        ON T.author_id = S.author_id
        WHEN MATCHED THEN
          UPDATE SET
            T.author_name = S.author_name,
            T.last_updated = CURRENT_TIMESTAMP()
        WHEN NOT MATCHED THEN
          INSERT (author_id, author_name, last_updated)
          VALUES (S.author_id, S.author_name, CURRENT_TIMESTAMP());
        """,
        # fact_video_statistics
        """
        INSERT INTO `adrineto-qst882-fall25.youtube_staging.fact_video_statistics`
        (video_id, date, view_count, like_count, comment_count)
        SELECT
          video_id,
          CURRENT_DATE() AS date,
          view_count,
          like_count,
          comment_count
        FROM `adrineto-qst882-fall25.youtube_raw.video_statistics`;
        """,
        # fact_comments
        """
        MERGE `adrineto-qst882-fall25.youtube_staging.fact_comments` AS T
        USING (
          SELECT
            comment_id,
            video_id,
            author_id,
            text_display AS comment_text,
            published_at
          FROM `adrineto-qst882-fall25.youtube_raw.comments`
        ) AS S
        ON T.comment_id = S.comment_id
        WHEN NOT MATCHED THEN
          INSERT (comment_id, video_id, author_id, comment_text, published_at)
          VALUES (S.comment_id, S.video_id, S.author_id, S.comment_text, S.published_at);
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
