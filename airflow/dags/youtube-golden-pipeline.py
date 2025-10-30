from airflow.decorators import dag, task
from datetime import datetime
from airflow.operators.python import get_current_context
from airflow.sensors.external_task import ExternalTaskSensor
import requests

# ----------------------------------------------------------------
# Helper function to call your deployed Cloud Function
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
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["youtube", "pipeline", "golden"],
)
def youtube_golden_pipeline():
    
    # Wait for the ETL DAG to finish
    wait_for_etl = ExternalTaskSensor(
        task_id="wait_for_elt", 
        external_dag_id="youtube_pipeline",
        external_task_id=None, 
        poke_interval=300,    
        timeout=3600 * 3,    
        mode="reschedule"
    )

    # Golden layer builder
    @task
    def golden(payload: dict = {}):
        url = "https://us-central1-adrineto-qst882-fall25.cloudfunctions.net/raw-golden"
        ctx = get_current_context()
        payload['date'] = ctx["ds_nodash"]
        resp = invoke_function(url, data=payload)
        print("Load Response:", resp)
        return resp

    # Task dependency
    wait_for_elt >> golden()

# Register DAG
youtube_golden_pipeline()



