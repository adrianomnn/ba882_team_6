import json
import os
from pathlib import Path
from datetime import datetime, timezone

import functions_framework
import duckdb
from google.cloud import secretmanager, storage

PROJECT_ID = "qst-ba-882-adam"
VERSION_ID = "latest"
DB = "youtube"
SCHEMA = "raw"
DB_SCHEMA = f"{DB}.{SCHEMA}"

RAW_BUCKET = os.environ.get("RAW_BUCKET")  # 与 extract 使用同一 bucket

def read_secret(project_id: str, secret_id: str, version: str = "latest") -> str:
    sm = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    resp = sm.access_secret_version(request={"name": name})
    return resp.payload.data.decode("utf-8")

def gcs_download(bucket_name: str, src: str, dst: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(src)
    Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(dst)
    return dst

@functions_framework.http
def task(request):
    """
    HTTP Params：
      table: one of [videos, video_stats, channels, categories, comments]
      run_id: extract 产出的 run_id
    落盘路径约定：
      gs://<bucket>/extract/run_id=<run_id>/table=<table>.jsonl
    """
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
    else:
        payload = request.args

    table = (payload.get("table") or "").strip().lower()
    run_id = (payload.get("run_id") or "").strip()

    if not RAW_BUCKET:
        return (json.dumps({"error": "Missing env RAW_BUCKET"}), 400, {"Content-Type": "application/json"})
    if not table or not run_id:
        return (json.dumps({"error": "table and run_id are required"}), 400, {"Content-Type": "application/json"})

    gcs_rel = f"extract/run_id={run_id}/table={table}.jsonl"
    local_path = f"/tmp/{table}-{run_id}.jsonl"

    try:
        # 1) 下载 JSONL
        gcs_download(RAW_BUCKET, gcs_rel, local_path)

        # 2) 读取 MotherDuck token 并连接
        md_token = read_secret(PROJECT_ID, "MotherDuck", VERSION_ID)
        con = duckdb.connect(f"md:?motherduck_token={md_token}")

        # 3) 读 JSON → 临时视图
        con.execute(f"""
            CREATE OR REPLACE VIEW tmp_{table} AS
            SELECT * FROM read_json_auto('{local_path}');
        """)

        # 4) INSERT 映射（针对五张 RAW 表分别写）
        if table == "videos":
            con.execute(f"""
                INSERT INTO {DB_SCHEMA}.videos
                SELECT
                    video_id::VARCHAR,
                    channel_id::VARCHAR,
                    title::VARCHAR,
                    description::VARCHAR,
                    TRY_CAST(published_at AS TIMESTAMP) AS published_at,
                    channel_title::VARCHAR,
                    search_query::VARCHAR,
                    TRY_CAST(ingest_timestamp AS TIMESTAMP) AS ingest_timestamp,
                    source_path::VARCHAR,
                    run_id::VARCHAR
                FROM tmp_videos;
            """)

        elif table == "video_stats":
            con.execute(f"""
                INSERT INTO {DB_SCHEMA}.video_stats
                SELECT
                    video_id::VARCHAR,
                    title::VARCHAR,
                    category_id::VARCHAR,
                    duration_iso8601::VARCHAR,
                    COALESCE(views::BIGINT, 0) AS views,
                    COALESCE(likes::BIGINT, 0) AS likes,
                    COALESCE(comments::BIGINT, 0) AS comments,
                    COALESCE(favorite_count::BIGINT, 0) AS favorite_count,
                    TRY_CAST(collected_at_utc AS TIMESTAMP) AS collected_at_utc,
                    TRY_CAST(ingest_timestamp AS TIMESTAMP) AS ingest_timestamp,
                    source_path::VARCHAR,
                    run_id::VARCHAR
                FROM tmp_video_stats;
            """)

        elif table == "channels":
            con.execute(f"""
                INSERT INTO {DB_SCHEMA}.channels
                SELECT
                    channel_id::VARCHAR,
                    channel_title::VARCHAR,
                    country::VARCHAR,
                    TRY_CAST(published_at AS TIMESTAMP) AS published_at,
                    COALESCE(subscriber_count::BIGINT, 0) AS subscriber_count,
                    COALESCE(video_count::BIGINT, 0) AS video_count,
                    COALESCE(view_count::BIGINT, 0) AS view_count,
                    TRY_CAST(ingest_timestamp AS TIMESTAMP) AS ingest_timestamp,
                    source_path::VARCHAR,
                    run_id::VARCHAR
                FROM tmp_channels;
            """)

        elif table == "categories":
            con.execute(f"""
                INSERT INTO {DB_SCHEMA}.categories
                SELECT
                    category_id::VARCHAR,
                    category_title::VARCHAR,
                    CAST(assignable AS BOOLEAN) AS assignable,
                    region::VARCHAR,
                    TRY_CAST(ingest_timestamp AS TIMESTAMP) AS ingest_timestamp,
                    source_path::VARCHAR,
                    run_id::VARCHAR
                FROM tmp_categories;
            """)

        elif table == "comments":
            con.execute(f"""
                INSERT INTO {DB_SCHEMA}.comments
                SELECT
                    video_id::VARCHAR,
                    comment_id::VARCHAR,
                    author::VARCHAR,
                    text::VARCHAR,
                    COALESCE(like_count::BIGINT, 0) AS like_count,
                    TRY_CAST(published_at AS TIMESTAMP) AS published_at,
                    TRY_CAST(updated_at AS TIMESTAMP) AS updated_at,
                    TRY_CAST(ingest_timestamp AS TIMESTAMP) AS ingest_timestamp,
                    source_path::VARCHAR,
                    run_id::VARCHAR
                FROM tmp_comments;
            """)
        else:
            return (json.dumps({"error": "unsupported table"}), 400, {"Content-Type": "application/json"})

        # 5) 返回结果
        stats = con.execute(f"SELECT COUNT(*) FROM {DB_SCHEMA}.{table}").fetchone()[0]
        return (json.dumps({
            "message": f"ingest ok: {table}",
            "gcs_src": f"gs://{RAW_BUCKET}/{gcs_rel}",
            "table_fqdn": f"{DB_SCHEMA}.{table}",
            "total_rows_in_table": stats
        }, ensure_ascii=False), 200, {"Content-Type": "application/json"})

    except Exception as e:
        return (json.dumps({"error": str(e)}), 500, {"Content-Type": "application/json"})
