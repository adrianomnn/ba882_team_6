from airflow.decorators import dag, task
from datetime import datetime, timedelta
from airflow.operators.python import get_current_context
import requests
import time
import threading

CLOUD_FUNCTION_URL = "https://us-central1-adrineto-qst882-fall25.cloudfunctions.net/mlops-pipeline-parallel"

def invoke_with_heartbeat(url: str, payload: dict):
    """Invoke with heartbeat to prevent timeout"""
    training_complete = threading.Event()
    
    def heartbeat():
        counter = 0
        while not training_complete.is_set():
            time.sleep(20)
            counter += 20
            if not training_complete.is_set():
                print(f"[{payload['algorithm']}] Still training... {counter}s")
    
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    
    try:
        print(f"Starting: {payload['algorithm']}")
        resp = requests.post(url, json=payload, timeout=600)
        training_complete.set()
        
        if resp.status_code == 200:
            result = resp.json()
            print(f"Success: {payload['algorithm']}")
            return result
        else:
            print(f"Failed: {payload['algorithm']} - Status {resp.status_code}")
            return {"status": "failed", "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        training_complete.set()
        print(f"Error: {payload['algorithm']} - {e}")
        return {"status": "failed", "error": str(e)}


@dag(
    schedule="@daily",
    start_date=datetime(2025, 11, 10),
    catchup=False,
    tags=["youtube", "ml"],
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
)
def youtube_train_simple():
    """Simple parallel training without dynamic mapping"""
    
    @task(execution_timeout=timedelta(minutes=12))
    def train_rf():
        ctx = get_current_context()
        payload = {
            "algorithm": "random_forest",
            "hyperparameters": {"n_estimators": 200, "max_depth": 10},
            "run_id": f"{ctx['dag_run'].run_id}_rf",
            "snapshot_date": ctx["ds"],
            "limit": 10000
        }
        return invoke_with_heartbeat(CLOUD_FUNCTION_URL, payload)
    
    @task(execution_timeout=timedelta(minutes=12))
    def train_gb():
        ctx = get_current_context()
        payload = {
            "algorithm": "gradient_boosting",
            "hyperparameters": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 5},
            "run_id": f"{ctx['dag_run'].run_id}_gb",
            "snapshot_date": ctx["ds"],
            "limit": 10000
        }
        return invoke_with_heartbeat(CLOUD_FUNCTION_URL, payload)
    
    @task(execution_timeout=timedelta(minutes=12))
    def train_lr():
        ctx = get_current_context()
        payload = {
            "algorithm": "logistic_regression",
            "hyperparameters": {"C": 1.0, "solver": "lbfgs"},
            "run_id": f"{ctx['dag_run'].run_id}_lr",
            "snapshot_date": ctx["ds"],
            "limit": 10000
        }
        return invoke_with_heartbeat(CLOUD_FUNCTION_URL, payload)
    
    # Execute in parallel
    rf_result = train_rf()
    gb_result = train_gb()
    lr_result = train_lr()

youtube_train_simple()