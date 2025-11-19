"""
Select Best Model from BigQuery and Promote to Production in GCS
"""

import json
import re
from google.cloud import bigquery, storage


def select_and_deploy_best_model(request):
    request_json = request.get_json(silent=True) or {}

    project_id = request_json.get("project_id")
    dataset = request_json.get("dataset")
    table = request_json.get("table")
    production_bucket = request_json.get("production_bucket")
    production_prefix = request_json.get(
        "production_prefix", "models/youtube/production/model.pkl"
    )

    if not all([project_id, dataset, table, production_bucket]):
        return {"error": "Missing required parameters"}, 400

    bq = bigquery.Client(project=project_id)
    storage_client = storage.Client(project=project_id)

    # --------------------------------------------------------------------
    # Query Best Model (highest F1)
    # --------------------------------------------------------------------
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

    rows = list(bq.query(query))
    if not rows:
        return {"error": "No training runs found"}, 404

    best = rows[0]

    metrics_json = json.loads(best["metrics"])
    best_f1 = float(metrics_json.get("f1", 0.0))

    model_path = best["model_gcs_path"]
    match = re.match(r"gs://([^/]+)/(.+)", model_path)

    if not match:
        return {"error": f"Invalid model path: {model_path}"}, 400

    src_bucket_name, src_blob_path = match.groups()

    src_bucket = storage_client.bucket(src_bucket_name)
    src_blob = src_bucket.blob(src_blob_path)

    dst_bucket = storage_client.bucket(production_bucket)
    dst_blob = dst_bucket.blob(production_prefix)

    # --------------------------------------------------------------------
    # Copy Model to Production
    # --------------------------------------------------------------------
    dst_bucket.copy_blob(src_blob, dst_bucket, new_name=production_prefix)

    # --------------------------------------------------------------------
    # Save Metadata JSON
    # --------------------------------------------------------------------
    metadata_blob = dst_bucket.blob(
        production_prefix.replace("model.pkl", "metadata.json")
    )

    metadata_blob.upload_from_string(
        json.dumps(
            {
                "run_id": best["run_id"],
                "algorithm": best["algorithm"],
                "f1": best_f1,
                "source_path": model_path,
                "deployed_at": best["created_at"].isoformat(),
            },
            indent=4,
        ),
        content_type="application/json",
    )

    return {
        "selected_model": model_path,
        "f1": best_f1,
        "algorithm": best["algorithm"],
        "production_path": f"gs://{production_bucket}/{production_prefix}",
    }