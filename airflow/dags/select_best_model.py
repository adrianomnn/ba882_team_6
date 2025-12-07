from airflow.decorators import dag, task
from airflow.datasets import Dataset
from datetime import datetime, timedelta
import requests
import json
import os

# Dataset created by youtube_mlops DAG
TRAINING_COMPLETE = Dataset("gs://mlops/youtube/training_complete")

# URL of your deployed Cloud Function
CFN_URL = "https://us-central1-adrineto-qst882-fall25.cloudfunctions.net/select-best-model"


@dag(
    start_date=datetime(2025, 1, 1),
    schedule=[TRAINING_COMPLETE],
    catchup=False,
    tags=["youtube", "ml", "selection"],
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    max_active_runs=1,
)
def youtube_select_best_model():

    @task
    def call_select_best_model():
        payload = {
            "project_id": "adrineto-qst882-fall25",
            "dataset": "youtube_metadata",
            "table": "training_runs_parallel",
            "production_bucket": "adrineto-ba882-fall25-team-6",
            "production_prefix": "models/youtube/production/model.pkl",
        }

        resp = requests.post(CFN_URL, json=payload)
        resp.raise_for_status()
        return resp.json()

    call_select_best_model()


youtube_select_best_model()
