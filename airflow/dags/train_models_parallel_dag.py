"""
Airflow DAG for parallel ML model training
Uses dynamic task mapping for true parallelism
"""
from airflow.decorators import dag, task
from datetime import datetime, timedelta
from airflow.operators.python import get_current_context
import requests
import os
import json

# ----------------------------------------------------------------
CLOUD_FUNCTION_URL = os.getenv(
    "TRAIN_MODEL_CLOUD_FUNCTION_URL",
    "https://us-central1-adrineto-qst882-fall25.cloudfunctions.net/train_model"
)

MODEL_CONFIGS = [
    {"algorithm": "random_forest", "hyperparameters": {"n_estimators": 200, "max_depth": 10}, "name": "rf"},
    {"algorithm": "gradient_boosting", "hyperparameters": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 5}, "name": "gb"},
    {"algorithm": "logistic_regression", "hyperparameters": {"C": 1.0, "solver": "lbfgs"}, "name": "lr"}
]

def invoke_training_function(url: str, payload: dict, timeout: int = 900):
    """
    Call Cloud Function and WAIT for training to complete.
    Heartbeat thread keeps the worker log alive while waiting.
    """
    import threading, time
    try:
        print(f"Invoking training: {payload.get('algorithm')} | run_id={payload.get('run_id')}")
        training_complete = threading.Event()

        def send_heartbeat():
            counter = 0
            while not training_complete.is_set():
                time.sleep(30)
                if not training_complete.is_set():
                    counter += 30
                    print(f"Heartbeat: training running ({counter}s)")

        heartbeat = threading.Thread(target=send_heartbeat, daemon=True)
        heartbeat.start()

        resp = requests.post(url, json=payload, timeout=timeout)

        training_complete.set()
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        print("Training call timed out")
        return {"status": "timeout", "error": "request timeout"}
    except requests.exceptions.RequestException as e:
        print("Training request exception:", e)
        return {"status": "failed", "error": str(e)}


@dag(
    schedule=None, #"@daily"
    start_date=datetime(2025, 11, 10),
    catchup=False,
    tags=["youtube", "ml", "training"],
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
        "execution_timeout": timedelta(minutes=30),
    },
    max_active_runs=1,
)
def youtube_mlops():
    @task
    def prepare_training_metadata():
        ctx = get_current_context()
        dag_run_id = ctx["dag_run"].run_id
        # Use actual run timestamp as snapshot_date (YYYY-MM-DD)
        # ctx["ts"] is like "2025-11-16T15:59:00+00:00"
        ts = ctx.get("ts")
        if ts:
            snapshot_date = ts.split("T")[0]
        else:
            snapshot_date = ctx.get("ds")  # fallback
        configs = []
        for idx, mc in enumerate(MODEL_CONFIGS):
            run_id = f"{dag_run_id}_{mc['name']}_{idx}"
            cfg = {
                "algorithm": mc["algorithm"],
                "hyperparameters": mc["hyperparameters"],
                "run_id": run_id,
                "snapshot_date": snapshot_date,
                "limit": 10000,
                "config_name": mc["name"]
            }
            configs.append(cfg)
        print(f"Prepared {len(configs)} configs for snapshot_date={snapshot_date}")
        return configs

    @task
    def train_single_model(config: dict):
        import sys
        sys.stdout.flush()
        print("---- START TRAIN ----")
        print("config:", json.dumps(config))
        sys.stdout.flush()
        result = invoke_training_function(CLOUD_FUNCTION_URL, config, timeout=900)
        print("TRAIN RESULT:", result)
        sys.stdout.flush()
        minimal = {
            "status": result.get("status"),
            "run_id": result.get("run_id") or config.get("run_id"),
            "config_name": config.get("config_name"),
            "algorithm": config.get("algorithm"),
            "metrics": result.get("metrics", {})
        }
        print("---- END TRAIN ----")
        return minimal

    training_configs = prepare_training_metadata()
    training_results = train_single_model.expand(config=training_configs)

youtube_mlops()