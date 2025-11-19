from airflow.decorators import dag, task
from airflow.datasets import Dataset
from datetime import datetime, timedelta
import json, os, re
from google.cloud import bigquery, storage

# Dataset published by the training DAG
#TRAINING_COMPLETE = Dataset("gs://youtube/model_training_done")

# --------------------------------------------------------------------
# Helper function
# --------------------------------------------------------------------
def select_and_deploy_best_model_fn(
    project_id: str,
    dataset: str,
    table: str,
    production_bucket: str,
    production_prefix: str = "models/youtube/production/model.pkl",
):
    bq = bigquery.Client(project=project_id)
    storage_client = storage.Client(project=project_id)

    query = f"""
        SELECT 
            run_id,
            model_gcs_path,
            algorithm,
            metrics,
            created_at
        FROM `{project_id}.{dataset}.{table}`
        ORDER BY SAFE_CAST(JSON_VALUE(metrics, '$.f1') AS FLOAT64) DESC
        LIMIT 1
    """

    result = list(bq.query(query))
    if not result:
        raise ValueError("No model runs found.")

    best = result[0]
    metrics_json = json.loads(best.metrics)
    best_f1 = float(metrics_json.get("f1", 0.0))

    # Extract GCS bucket and path
    match = re.match(r"gs://([^/]+)/(.+)", best.model_gcs_path)
    if not match:
        raise ValueError(f"Invalid GCS path: {best.model_gcs_path}")

    src_bucket_name, src_blob_path = match.groups()

    src_bucket = storage_client.bucket(src_bucket_name)
    src_blob = src_bucket.blob(src_blob_path)

    dst_bucket = storage_client.bucket(production_bucket)
    dst_blob = dst_bucket.blob(production_prefix)

    # Copy to production
    dst_bucket.copy_blob(src_blob, dst_bucket, production_prefix)

    # Write metadata file
    metadata_blob = dst_bucket.blob(
        production_prefix.replace("model.pkl", "metadata.json")
    )
    metadata_blob.upload_from_string(
        json.dumps(
            {
                "run_id": best.run_id,
                "algorithm": best.algorithm,
                "f1": best_f1,
                "source_path": best.model_gcs_path,
                "deployed_at": best.created_at.isoformat(),
            },
            indent=4,
        ),
        content_type="application/json",
    )

    return {
        "selected_model": best.model_gcs_path,
        "f1": best_f1,
        "algorithm": best.algorithm,
    }


# --------------------------------------------------------------------
# DAG Task
# --------------------------------------------------------------------
@dag(
    start_date=datetime(2025, 1, 1),
    schedule=None, #[TRAINING_COMPLETE],
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
    def select_best_model():
        return select_and_deploy_best_model_fn(
            project_id="adrineto-qst882-fall25",
            dataset="youtube_metadata",
            table="training_runs_parallel",
            production_bucket="adrineto-ba882-fall25-team-6",
            production_prefix="models/youtube/production/model.pkl",
        )

    select_best_model()


youtube_select_best_model()
