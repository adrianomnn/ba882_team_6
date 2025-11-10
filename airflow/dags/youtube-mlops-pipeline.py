from airflow.decorators import dag, task
from airflow.sensors.external_task import ExternalTaskSensor
import requests
import os
import json

CLOUD_FUNC_URL = os.getenv("YOUTUBE_TRAIN_FUNC_URL", "https://us-central1-adrineto-qst882-fall25.cloudfunctions.net/mlops-pipeline")
DEFAULT_TIMEOUT = 60 * 60 * 2  # 2 hours

def call_cloud_function(url, payload=None):
    resp = requests.post(url, json=payload or {}, timeout=300)
    resp.raise_for_status()
    return resp.json()

@dag(
    schedule="@daily",
    start_date=datetime(2025, 11, 10),
    catchup=False,
    tags=["youtube", "mlops"]
)
def youtube_train_pipeline():

    # wait_for_etl = ExternalTaskSensor(
    #     task_id="wait_for_etl",
    #     external_dag_id="youtube_pipeline",
    #     external_task_id=None,
    #     poke_interval=300,
    #     timeout=3600 * 3,
    #     mode="reschedule"
    # )

    @task
    def invoke_training(snapshot_date=None):
        payload = {}
        if snapshot_date:
            payload["snapshot_date"] = snapshot_date
        # optional: add "limit" to restrict rows for tests
        result = call_cloud_function(CLOUD_FUNC_URL, payload)
        return result

    # you can pass DS or ds_nodash via templating if needed
    training_result = invoke_training()
    # wait_for_etl >> training_result

youtube_train_pipeline = youtube_train_pipeline()