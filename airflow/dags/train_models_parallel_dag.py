from airflow.decorators import dag, task
from datetime import datetime
from airflow.operators.python import get_current_context
import requests
import os

# ----------------------------------------------------------------
# Helper function to call your deployed Cloud Function
# ----------------------------------------------------------------
def invoke_function(url, data=None):
    """
    Invoke our training Cloud Function via POST with JSON payload.
    """
    resp = requests.post(url, json=data or {})
    resp.raise_for_status()
    return resp.json()

# ----------------------------------------------------------------
# DAG Definition
# ----------------------------------------------------------------
@dag(
    schedule=None,  # set to "@daily" or cron if needed
    start_date=datetime(2025, 11, 10),
    catchup=False,
    tags=["youtube", "ml", "training"]
)
def youtube_train_models_parallel():

    CLOUD_FUNCTION_URL = os.getenv(
        "TRAIN_MODEL_CLOUD_FUNCTION_URL",
        "https://us-central1-adrineto-qst882-fall25.cloudfunctions.net/mlops-pipeline-parallel"
    )

    # Default hyperparameters for each model
    MODEL_CONFIGS = {
        "random_forest": {"n_estimators": 200, "max_depth": 10},
        "gradient_boosting": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 5},
        "logistic_regression": {"C": 1.0, "solver": "lbfgs"}
    }

    # ----------------------------------------------------------------
    # Individual training tasks (parallel)
    # ----------------------------------------------------------------
    @task
    def train_random_forest():
        ctx = get_current_context()
        payload = {
            "algorithm": "random_forest",
            "hyperparameters": MODEL_CONFIGS["random_forest"],
            "snapshot_date": ctx["ds_nodash"], 
            "limit": 10000,
            "run_id": ctx["dag_run"].run_id,
        }
        resp = invoke_function(CLOUD_FUNCTION_URL, data=payload)
        print("Random Forest Training Response:", resp)
        return resp

    @task
    def train_gradient_boosting():
        ctx = get_current_context()
        payload = {
            "algorithm": "gradient_boosting",
            "hyperparameters": MODEL_CONFIGS["gradient_boosting"],
            "snapshot_date": ctx["ds_nodash"],
            "limit": 10000,
            "run_id": ctx["dag_run"].run_id,
        }
        resp = invoke_function(CLOUD_FUNCTION_URL, data=payload)
        print("Gradient Boosting Training Response:", resp)
        return resp

    @task
    def train_logistic_regression():
        ctx = get_current_context()
        payload = {
            "algorithm": "logistic_regression",
            "hyperparameters": MODEL_CONFIGS["logistic_regression"],
            "snapshot_date": ctx["ds_nodash"],
            "limit": 10000,
            "run_id": ctx["dag_run"].run_id,
        }
        resp = invoke_function(CLOUD_FUNCTION_URL, data=payload)
        print("Logistic Regression Training Response:", resp)
        return resp

    # ----------------------------------------------------------------
    # DAG task dependency graph
    # ----------------------------------------------------------------
    rf_result = train_random_forest()
    gb_result = train_gradient_boosting()
    lr_result = train_logistic_regression()

# Instantiate DAG
youtube_train_models_parallel()
