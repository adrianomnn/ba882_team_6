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
# Configuration
# ----------------------------------------------------------------
CLOUD_FUNCTION_URL = os.getenv(
    "TRAIN_MODEL_CLOUD_FUNCTION_URL",
    "https://us-central1-adrineto-qst882-fall25.cloudfunctions.net/mlops-pipeline-parallel"
)

# Model configurations - this will be expanded into parallel tasks
MODEL_CONFIGS = [
    {
        "algorithm": "random_forest",
        "hyperparameters": {"n_estimators": 200, "max_depth": 10},
        "name": "rf"
    },
    {
        "algorithm": "gradient_boosting",
        "hyperparameters": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 5},
        "name": "gb"
    },
    {
        "algorithm": "logistic_regression",
        "hyperparameters": {"C": 1.0, "solver": "lbfgs"},
        "name": "lr"
    }
]

# ----------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------
def invoke_training_function(url: str, payload: dict, timeout: int = 600):
    """
    Call Cloud Function and WAIT for training to complete.
    Uses longer timeout since training takes time.
    Sends periodic heartbeats to prevent Airflow timeout.
    """
    import threading
    import time

    try:
        print(f"Invoking training: {payload.get('algorithm')} | {payload.get('run_id')}")

        # Create a flag to stop heartbeat thread
        training_complete = threading.Event()

        # Heartbeat function - prints every 30 seconds to keep Airflow without errors
        def send_heartbeat():
            counter = 0
            while not training_complete.is_set():
                time.sleep(30)
                if not training_complete.is_set():
                    counter += 30
                    print(f"Heartbeat: Training still running... ({counter}s elapsed)")

        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
        heartbeat_thread.start()

        resp = requests.post(url, json=payload, timeout=timeout)

        # Stop heartbeat
        training_complete.set()

        resp.raise_for_status()

        result = resp.json()
        print(f"Training completed: {result.get('status')}")

        return result

    except requests.exceptions.Timeout:
        print(f"Training request timed out after {timeout}s")
        return {"status": "timeout", "error": "Request timeout"}

    except requests.exceptions.RequestException as e:
        print(f"Training request failed: {e}")
        return {"status": "failed", "error": str(e)}


# ----------------------------------------------------------------
# DAG Definition
# ----------------------------------------------------------------
@dag(
    schedule="@daily",
    start_date=datetime(2025, 11, 10),
    catchup=False,
    tags=["youtube", "ml", "training", "parallel"],
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
        "execution_timeout": timedelta(minutes=15),
    },
    max_active_runs=1,
)
def youtube_train_models_parallel():
    """
    Train multiple ML models in parallel using dynamic task mapping
    """

    # ----------------------------------------------------------------
    # Task 1: Prepare metadata for all training runs
    # ----------------------------------------------------------------
    @task
    def prepare_training_metadata():
        """
        Generate run IDs and metadata for all models
        Returns list of configs with unique run IDs
        """
        ctx = get_current_context()
        dag_run_id = ctx["dag_run"].run_id
        execution_date = ctx["ds"]

        configs_with_metadata = []

        for idx, config in enumerate(MODEL_CONFIGS):
            # Create unique run_id for each model
            run_id = f"{dag_run_id}_{config['name']}_{idx}"

            training_config = {
                "algorithm": config["algorithm"],
                "hyperparameters": config["hyperparameters"],
                "run_id": run_id,
                "snapshot_date": execution_date,
                "limit": 10000, 
                "config_name": config["name"]
            }

            configs_with_metadata.append(training_config)

        print(f"Prepared {len(configs_with_metadata)} training configurations")
        return configs_with_metadata

    # ----------------------------------------------------------------
    # Task 2: Train individual model
    # ----------------------------------------------------------------
    @task
    def train_single_model(config: dict):
        """
        Train a single model configuration.
        This task will be executed in parallel for each config.
        
        Returns minimal result to avoid XCom size issues.
        """
        import sys
        
        # Flush output frequently to ensure logs are captured
        sys.stdout.flush()
        sys.stderr.flush()
        
        print(f"{'='*60}")
        print(f"Starting training for: {config.get('config_name')}")
        print(f"Algorithm: {config.get('algorithm')}")
        print(f"Run ID: {config.get('run_id')}")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        result = invoke_training_function(
            CLOUD_FUNCTION_URL,
            config,
            timeout=600
        )

        # Add config info to result
        result["config_name"] = config.get("config_name")
        result["algorithm"] = config.get("algorithm")
        result["run_id"] = config.get("run_id")

        # Return minimal result to avoid XCom issues
        minimal_result = {
            "status": result.get("status"),
            "run_id": result.get("run_id"),
            "config_name": result.get("config_name"),
            "algorithm": result.get("algorithm"),
            "metrics": result.get("metrics", {}),
        }
        
        print(f"{'='*60}")
        print(f"Training completed: {minimal_result.get('status')}")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        return minimal_result

    # ----------------------------------------------------------------
    # Task Graph - Define dependencies
    # ----------------------------------------------------------------

    # Step 1: Prepare all configurations
    training_configs = prepare_training_metadata()

    # Step 2: Train models in parallel using expand()
    # This creates N parallel tasks, one for each config
    training_results = train_single_model.expand(config=training_configs)

# ----------------------------------------------------------------
# Instantiate DAG
# ----------------------------------------------------------------