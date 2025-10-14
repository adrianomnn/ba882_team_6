from airflow.decorators import dag, task
from datetime import datetime
from airflow.operators.python import get_current_context
import requests

# ----------------------------------------------------------------
# Helper function to call your deployed Cloud Functions
# ----------------------------------------------------------------
def invoke_function(url, data=None):
    """
    Invoke our cloud function via POST with JSON payload.
    """
    resp = requests.post(url, json=data or {})
    resp.raise_for_status()
    return resp.json()

# ----------------------------------------------------------------
# DAG definition
# ----------------------------------------------------------------
@dag(
    schedule="@daily",   # Runs once a day at midnight UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["youtube", "pipeline"]
)
def youtube_pipeline():
    
    # STEP 1 - Create schema (BigQuery table)
    @task
    def schema():
        url = "https://us-central1-adrineto-qst882-fall25.cloudfunctions.net/raw-schema"
        resp = invoke_function(url)
        print("Schema Response:", resp)
        return resp

    # STEP 2 - Extract data from YouTube API
    @task
    def extract(payload: dict):
        url = "https://us-central1-adrineto-qst882-fall25.cloudfunctions.net/raw-extract"
        ctx = get_current_context()
        payload['run_id'] = ctx["dag_run"].run_id
        payload['date'] = ctx["ds_nodash"]
        resp = invoke_function(url, data=payload)
        print("Extract Response:", resp)
        return resp

    # STEP 3 - Load data to BigQuery
    @task
    def load(payload: dict):
        url = "https://us-central1-adrineto-qst882-fall25.cloudfunctions.net/raw-parse"
        ctx = get_current_context()
        payload['date'] = ctx["ds_nodash"]
        resp = invoke_function(url, data=payload)
        print("Load Response:", resp)
        return resp

    # STEP 4 - Transform data to BigQuery
    @task
    def transform(payload: dict):
        url = "https://us-central1-adrineto-qst882-fall25.cloudfunctions.net/raw-transform"
        ctx = get_current_context()
        payload['date'] = ctx["ds_nodash"]
        resp = invoke_function(url, data=payload)
        print("Load Response:", resp)
        return resp

    # Define task dependencies (schema → extract → load)
    schema_result = schema()
    extract_result = extract(schema_result)
    load_result = load(extract_result)
    transform_result = transform(parse_result)

youtube_pipeline()