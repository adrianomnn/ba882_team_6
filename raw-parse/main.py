"""
Load YouTube data from GCS to BigQuery
"""
import functions_framework
from google.cloud import storage, bigquery
import pandas as pd
import json
from datetime import datetime

project_id = 'adrineto-qst882-fall25'
dataset_id = 'youtube_raw'

@functions_framework.http
def task(request):
    """
    Cloud Function to load YouTube data from GCS to BigQuery
    """
    try:
        # Parse request
        request_json = request.get_json(silent=True)
        if request_json is None:
            return {"error": "Missing JSON payload"}, 400

        bucket_name = request_json.get("bucket_name")
        blob_name = request_json.get("blob_name")
        run_id = request_json.get("run_id")

        if not all([bucket_name, blob_name, run_id]):
            return {"error": "Missing required fields: bucket_name, blob_name, run_id"}, 400

        print(f"Loading data from gs://{bucket_name}/{blob_name}")

        # Download data from GCS
        print("Downloading data from GCS...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        data_str = blob.download_as_text()
        data = json.loads(data_str)

        # Convert JSON to DataFrames
        print("Converting to DataFrames...")
        videos_df = pd.DataFrame(data.get("videos", []))
        channels_df = pd.DataFrame(data.get("channels", []))
        comments_df = pd.DataFrame(data.get("comments", []))
        video_stats_df = pd.DataFrame(data.get("video_stats", []))
        categories_df = pd.DataFrame(data.get("categories", []))

        # Add metadata columns
        ingest_ts = datetime.utcnow()
        for df in [videos_df, channels_df, categories_df, comments_df, video_stats_df]:
            if not df.empty:
                df["ingest_timestamp"] = ingest_ts
                df["source_path"] = f"gs://{bucket_name}/{blob_name}"
                df["run_id"] = run_id

        # Initialize BigQuery client
        print("Connecting to BigQuery...")
        bq_client = bigquery.Client(project=project_id)

        # Load configuration
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            autodetect=False
        )

        rows_inserted = 0

        # Insert videos
        if not videos_df.empty:
            print(f"Inserting {len(videos_df)} videos...")
            table_id = f"{project_id}.{dataset_id}.videos"
            job = bq_client.load_table_from_dataframe(
                videos_df, table_id, job_config=job_config
            )
            job.result()  # Wait for completion
            rows_inserted += len(videos_df)
            print(f"Loaded {len(videos_df)} videos")

        # Insert channels
        if not channels_df.empty:
            print(f"Inserting {len(channels_df)} channels...")
            table_id = f"{project_id}.{dataset_id}.channels"
            job = bq_client.load_table_from_dataframe(
                channels_df, table_id, job_config=job_config
            )
            job.result()
            rows_inserted += len(channels_df)
            print(f"Loaded {len(channels_df)} channels")
        
        # Insert comments
        if not comments_df.empty:
            print(f"Inserting {len(comments_df)} comments...")
            table_id = f"{project_id}.{dataset_id}.comments"
            job = bq_client.load_table_from_dataframe(
                comments_df, table_id, job_config=job_config
            )
            job.result()
            rows_inserted += len(comments_df)
            print(f"Loaded {len(comments_df)} comments")

        # Insert video stats
        if not video_stats_df.empty:
            print(f"Inserting {len(video_stats_df)} video stats...")
            table_id = f"{project_id}.{dataset_id}.video_statistics"
            job = bq_client.load_table_from_dataframe(
                video_stats_df, table_id, job_config=job_config
            )
            job.result()
            rows_inserted += len(video_stats_df)
            print(f"Loaded {len(video_stats_df)} video stats")

        # Insert categories
        if not categories_df.empty:
            print(f"Inserting {len(categories_df)} categories...")
            table_id = f"{project_id}.{dataset_id}.categories"
            job = bq_client.load_table_from_dataframe(
                categories_df, table_id, job_config=job_config
            )
            job.result()
            rows_inserted += len(categories_df)
            print(f"Loaded {len(categories_df)} categories")

        result = {
            "status": "success",
            "run_id": run_id,
            "rows_inserted": rows_inserted,
            "tables_updated": {
                "videos": len(videos_df),
                "channels": len(channels_df),
                "comments": len(comments_df),
                "video_stats": len(video_stats_df),
                "categories": len(categories_df)
            },
            "destination": f"{project_id}.{dataset_id}"
        }

        print(f"Data successfully loaded to BigQuery: {result}")
        return result, 200

    except Exception as e:
        error_msg = f"Load failed: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}, 500