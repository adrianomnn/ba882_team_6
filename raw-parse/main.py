"""
Load YouTube data from GCS to BigQuery
"""
# load_youtube.py
import functions_framework
from google.cloud import storage, bigquery
import pandas as pd
import json
from datetime import datetime

project_id = 'adrineto-qst882-fall25'
dataset_id = 'youtube_raw'

@functions_framework.http
def task(request):
    request_json = request.get_json(silent=True)
    if request_json is None:
        return {"status": "failed", "error": "Missing payload"}, 400

    bucket_name = request_json["bucket_name"]
    blob_name = request_json["blob_name"]
    run_id = request_json["run_id"]

    # Access data from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data_str = blob.download_as_text()
    data = json.loads(data_str)

    # Convert JSON to DataFrames
    videos_df = pd.DataFrame(data.get("videos", []))
    channels_df = pd.DataFrame(data.get("channels", []))
    comments_df = pd.DataFrame(data.get("comments", []))
    video_stats_df = pd.DataFrame(data.get("video_stats", []))
    categories_df = pd.DataFrame(data.get("categories", []))

    ingest_ts = datetime.utcnow()
    for df in [videos_df, channels_df, comments_df, video_stats_df, categories_df]:
        if not df.empty:
            df["ingest_timestamp"] = ingest_ts
            df["source_path"] = f"gs://{bucket_name}/{blob_name}"
            df["run_id"] = run_id

    # Connect to BigQuery
    bq_client = bigquery.Client(project=project_id)

    def load_table(df, table_name):
        if df.empty:
            print(f"Skipping {table_name} (empty DataFrame)")
            return
        
        for col in ["published_at", "updated_at", "created_at", "collected_at"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        table_id = f"{project_id}.{dataset_id}.{table_name}"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        job = bq_client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
        print(f"Loaded {len(df)} rows into {table_id}")

    # Load each table
    load_table(videos_df, "videos")
    load_table(channels_df, "channels")
    load_table(comments_df, "comments")
    load_table(video_stats_df, "video_statistics")
    load_table(categories_df, "categories")

    return {
        "status": "success",
        "message": "Data loaded successfully to BigQuery",
        "project": project_id,
        "dataset": dataset_id,
        "run_id": run_id
    }, 200
