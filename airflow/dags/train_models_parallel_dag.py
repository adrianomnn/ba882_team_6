from airflow.decorators import dag, task
from datetime import datetime, timedelta
from airflow.operators.python import get_current_context
import requests
import os
import time

# ----------------------------------------------------------------
# Helper function with timeout + retries + heartbeat-safe logging
# ----------------------------------------------------------------
def invoke_function(url, data=None):
    MAX_RETRIES = 3
    TIMEOUT_SEC = 10  # short timeout to avoid blocking worker

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[INFO] Calling Cloud Function (attempt {attempt})...")
            resp = requests.post(url, json=data or {}, timeout=TIMEOUT_SEC)
            resp.raise_for_status()
            return resp.json()

        except requests.Timeout:
            print(f"[WARN] Timeout calling Cloud Function on attempt {attempt}/{MAX_RETRIES}")
        except Exception as e:
            print(f"[ERROR] Error calling Cloud Function: {e}")

        time.sleep(2)

    raise RuntimeError("Cloud Function failed after max retries")

# ----------------------------------------------------------------
# DAG Definition
# ----------------------------------------------------------------
@dag(
    schedule="@daily",
    start_date=datetime(2025, 11, 10),
    catchup=False,
    tags=["youtube", "ml", "training"],
        default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=15),
        "task_timeout": timedelta(minutes=5),  # prevents heartbeat timeout
    }
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
    def train_model(algorithm_name: str):
        ctx = get_current_context()
        formatted_date = datetime.strptime(ctx["ds_nodash"], "%Y%m%d").strftime("%Y-%m-%d")

        payload = {
            "algorithm": algorithm_name,
            "hyperparameters": MODEL_CONFIGS[algorithm_name],
            "snapshot_date": formatted_date,
            "limit": 10000,
            "run_id": ctx["dag_run"].run_id,
        }

        print(f"[INFO] Triggering training for {algorithm_name} at {formatted_date}")

        response = invoke_function(CLOUD_FUNCTION_URL, data=payload)

        print(f"[INFO] Response for {algorithm_name}:", response)
        return response
    
    # ----------------------------------------------------------------
    # DAG task dependency graph
    # ----------------------------------------------------------------
    # Launch 3 parallel tasks
    rf = train_model("random_forest")
    gb = train_model("gradient_boosting")
    lr = train_model("logistic_regression")


# Instantiate DAG
youtube_train_models_parallel()
