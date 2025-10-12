# raw_schema_youtube.py
import functions_framework
from google.cloud import secretmanager
import duckdb

# settings
project_id = 'adrineto-qst882-fall25'
secret_id = 'MotherDuck'
version_id = 'latest'

# db setup
db = 'youtube'
schema = "raw"
db_schema = f"{db}.{schema}"

@functions_framework.http
def task(request):

    sm = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = sm.access_secret_version(request={"name": name})
    md_token = response.payload.data.decode("UTF-8")

    md = duckdb.connect(f"md:?motherduck_token={md_token}")

    # Create DB and schema
    md.sql(f"CREATE DATABASE IF NOT EXISTS {db};")
    md.sql(f"CREATE SCHEMA IF NOT EXISTS {db_schema};")
    print(f"Database and schema ensured: {db_schema}")

    # Videos table
    md.sql(f"""
        CREATE TABLE IF NOT EXISTS {db_schema}.videos (
            video_id VARCHAR,
            channel_id VARCHAR,
            title VARCHAR,
            description VARCHAR,
            published_at TIMESTAMP,
            search_query VARCHAR,
            search_order VARCHAR,
            ingest_timestamp TIMESTAMP,
            source_path VARCHAR,
            run_id VARCHAR
        );
    """)

    # Channels table
    md.sql(f"""
        CREATE TABLE IF NOT EXISTS {db_schema}.channels (
            channel_id VARCHAR,
            title VARCHAR,
            description VARCHAR,
            subscribers_count BIGINT,
            video_count BIGINT,
            view_count BIGINT,
            published_at TIMESTAMP,
            country VARCHAR,
            ingest_timestamp TIMESTAMP,
            source_path VARCHAR,
            run_id VARCHAR
        );
    """)

    # Video statistics table
    md.sql(f"""
        CREATE TABLE IF NOT EXISTS {db_schema}.video_statistics (
            video_id STRING PRIMARY KEY,
            category_id STRING,
            tags STRING,
            duration STRING,
            view_count INTEGER,
            like_count INTEGER,
            comment_count INTEGER,
            favorite_count INTEGER,
            ingest_timestamp TIMESTAMP,
            source_path VARCHAR,
            run_id VARCHAR
        );
    """)

    # Comments table
    md.sql(f"""
        CREATE TABLE IF NOT EXISTS {db_schema}.comments (
            comment_id STRING PRIMARY KEY,
            video_id STRING,
            author_display_name STRING,
            text_display STRING,
            like_count INTEGER,
            published_at TIMESTAMP,
            ingest_timestamp TIMESTAMP,
            source_path VARCHAR,
            run_id VARCHAR
        );
    """)

    # Categories table
    md.sql(f"""
        CREATE TABLE IF NOT EXISTS {db_schema}.categories (
            category_id STRING PRIMARY KEY,
            title STRING,
            assignable BOOLEAN,
            ingest_timestamp TIMESTAMP,
            source_path VARCHAR,
            run_id VARCHAR
        );
    """)

    print("Schema and tables created in MotherDuck.")
    return {}, 200
