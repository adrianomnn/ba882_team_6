from src.youtube_api import (
    get_channel_details, get_videos,
    get_video_statistics, get_video_comments, get_video_categories
)
from src.load import load_to_bigquery
from src.schema import create_youtube_tables
import os

def run_youtube_pipeline(query="data engineering", request=None):
    project_id = os.getenv("GCP_PROJECT", "adrineto-qst882-fall25")
    dataset_id = os.getenv("DATASET_ID", "youtube_data")

    create_youtube_tables(project_id, dataset_id)

    videos_df = get_videos(query, max_results=10)

    channels_df = get_channel_details(videos_df["channel_id"].dropna().unique().tolist())
    stats_df = get_video_statistics(videos_df["video_id"].tolist())
    comments_df = get_video_comments(videos_df["video_id"].iloc[0]) # just the comments from the most relevant video
    categories_df = get_video_categories(region_code="US")

    load_to_bigquery(channels_df, f"{project_id}.{dataset_id}.channels")
    load_to_bigquery(videos_df, f"{project_id}.{dataset_id}.videos")
    load_to_bigquery(stats_df, f"{project_id}.{dataset_id}.video_statistics")
    load_to_bigquery(comments_df, f"{project_id}.{dataset_id}.comments")
    load_to_bigquery(categories_df, f"{project_id}.{dataset_id}.video_categories")

    return "YouTube ETL pipeline executed successfully!"
