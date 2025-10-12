# load_youtube.py
import functions_framework
from google.cloud import secretmanager, storage
import duckdb
import pandas as pd
import json
from datetime import datetime

project_id = 'adrineto-qst882-fall25'
secret_id = 'MotherDuck'
version_id = 'latest'
db = 'youtube'
schema = 'raw'
db_schema = f"{db}.{schema}"

@functions_framework.http
def task(request):
    request_json = request.get_json(silent=True)
    if request_json is None:
        return {"error": "Missing payload"}, 400

    bucket_name = request_json["bucket_name"]
    blob_name = request_json["blob_name"]
    run_id = request_json["run_id"]

    # Access secrets
    sm = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = sm.access_secret_version(request={"name": name})
    md_token = response.payload.data.decode("UTF-8")

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

    for df in [videos_df, channels_df, categories_df, comments_df, video_stats_df]:
        df["ingest_timestamp"] = ingest_ts
        df["source_path"] = f"gs://{bucket_name}/{blob_name}"
        df["run_id"] = run_id

    # Connect to MotherDuck
    md = duckdb.connect(f"md:?motherduck_token={md_token}")

    # Insert into MotherDuck tables
    if not videos_df.empty:
        md.execute(f"INSERT INTO {db_schema}.videos SELECT * FROM videos_df")

    if not channels_df.empty:
        md.execute(f"INSERT INTO {db_schema}.channels SELECT * FROM channels_df")
    
    if not comments_df.empty:
        md.execute(f"INSERT INTO {db_schema}.comments SELECT * FROM comments_df")

    if not video_stats_df.empty:
        md.execute(f"INSERT INTO {db_schema}.video_statistics SELECT * FROM video_stats_df")

    if not categories_df.empty:
        md.execute(f"INSERT INTO {db_schema}.categories SELECT * FROM categories_df")

    md.close()

    print("Data successfully loaded into MotherDuck.")
    return {}, 200
