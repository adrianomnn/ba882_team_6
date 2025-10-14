import json
import functions_framework
from google.cloud import secretmanager
import duckdb

# ---------- settings ----------
project_id = 'qst-ba-882-adam'
version_id = 'latest'
db       = 'youtube'
schema   = 'raw'
db_schema = f'{db}.{schema}'

def _read_secret(client, name):
    resp = client.access_secret_version(request={"name": name})
    return resp.payload.data.decode("UTF-8")

@functions_framework.http
def task(request):
    status = {"secrets": {}, "ddl": "pending"}

    # 1) 读取两个 Secret（不会回显明文）
    sm = secretmanager.SecretManagerServiceClient()
    md_name = f"projects/{project_id}/secrets/MotherDuck/versions/{version_id}"
    yt_name = f"projects/{project_id}/secrets/YOUTUBE_API_KEY/versions/{version_id}"

    try:
        md_token = _read_secret(sm, md_name)
        status["secrets"]["MotherDuck"] = {
            "ok": True,
            "len": len(md_token),
            "looks_like_md_token": md_token.startswith("md_")
        }
    except Exception as e:
        status["secrets"]["MotherDuck"] = {"ok": False, "error": str(e)}

    try:
        yt_key = _read_secret(sm, yt_name)
        status["secrets"]["YOUTUBE_API_KEY"] = {
            "ok": True,
            "len": len(yt_key),
            "prefix": yt_key[:4]  
        }
    except Exception as e:
        status["secrets"]["YOUTUBE_API_KEY"] = {"ok": False, "error": str(e)}

    # 2) 
    try:
        if status["secrets"].get("MotherDuck", {}).get("ok"):
            md = duckdb.connect(f"md:?motherduck_token={md_token}")

            md.sql(f"CREATE DATABASE IF NOT EXISTS {db};")
            md.sql(f"CREATE SCHEMA IF NOT EXISTS {db_schema};")

            md.sql(f"""
            CREATE TABLE IF NOT EXISTS {db_schema}.channels (
                channel_id         VARCHAR,
                channel_title      VARCHAR,
                country            VARCHAR,
                published_at       TIMESTAMP,
                subscriber_count   BIGINT,
                video_count        BIGINT,
                view_count         BIGINT,
                ingest_timestamp   TIMESTAMP,
                source_path        VARCHAR,
                run_id             VARCHAR
            );
            """)

            md.sql(f"""
            CREATE TABLE IF NOT EXISTS {db_schema}.videos (
                video_id           VARCHAR,
                channel_id         VARCHAR,
                title              VARCHAR,
                description        VARCHAR,
                published_at       TIMESTAMP,
                channel_title      VARCHAR,
                search_query       VARCHAR,
                ingest_timestamp   TIMESTAMP,
                source_path        VARCHAR,
                run_id             VARCHAR
            );
            """)

            md.sql(f"""
            CREATE TABLE IF NOT EXISTS {db_schema}.video_stats (
                video_id           VARCHAR,
                title              VARCHAR,
                category_id        VARCHAR,
                duration_iso8601   VARCHAR,
                views              BIGINT,
                likes              BIGINT,
                comments           BIGINT,
                favorite_count     BIGINT,
                collected_at_utc   TIMESTAMP,
                ingest_timestamp   TIMESTAMP,
                source_path        VARCHAR,
                run_id             VARCHAR
            );
            """)

            md.sql(f"""
            CREATE TABLE IF NOT EXISTS {db_schema}.comments (
                video_id           VARCHAR,
                comment_id         VARCHAR,
                author             VARCHAR,
                text               VARCHAR,
                like_count         BIGINT,
                published_at       TIMESTAMP,
                updated_at         TIMESTAMP,
                ingest_timestamp   TIMESTAMP,
                source_path        VARCHAR,
                run_id             VARCHAR
            );
            """)

            md.sql(f"""
            CREATE TABLE IF NOT EXISTS {db_schema}.categories (
                category_id        VARCHAR,
                category_title     VARCHAR,
                assignable         BOOLEAN,
                region             VARCHAR,
                ingest_timestamp   TIMESTAMP,
                source_path        VARCHAR,
                run_id             VARCHAR
            );
            """)
            status["ddl"] = "ok"
        else:
            status["ddl"] = "skipped_no_motherduck_secret"
    except Exception as e:
        status["ddl"] = f"error: {e}"

    return (json.dumps({
        "message": f"Created/verified RAW schema at {db_schema}",
        "status": status
    }, ensure_ascii=False), 200, {"Content-Type": "application/json"})
