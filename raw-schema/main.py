"""
Create BigQuery dataset and tables for YouTube data
"""
import functions_framework
from google.cloud import bigquery

project_id = 'adrineto-qst882-fall25'
dataset_id = 'youtube_raw'

@functions_framework.http
def task(request):
    """
    Create or update YouTube database schema in BigQuery
    """
    try:
        # Get request parameters
        request_json = request.get_json(silent=True)
        drop_existing = False
        
        if request_json:
            drop_existing = request_json.get('drop_existing', False)
        
        if request.args:
            drop_existing = request.args.get('drop_existing', 'false').lower() == 'true'
        
        print(f"Setting up BigQuery schema (drop_existing={drop_existing})")
        
        # Initialize BigQuery client
        client = bigquery.Client(project=project_id)
        
        # Create dataset if not exists
        dataset_ref = f"{project_id}.{dataset_id}"
        try:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "us-central1"
            dataset = client.create_dataset(dataset, exists_ok=True)
            print(f"Dataset {dataset_id} ready")
        except Exception as e:
            print(f"Dataset already exists or error: {e}")
        
        # Define table schemas
        tables_config = {
            'videos': [
                bigquery.SchemaField("video_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("channel_id", "STRING"),
                bigquery.SchemaField("title", "STRING"),
                bigquery.SchemaField("description", "STRING"),
                bigquery.SchemaField("published_at", "TIMESTAMP"),
                bigquery.SchemaField("search_query", "STRING"),
                bigquery.SchemaField("search_order", "STRING"),
                bigquery.SchemaField("ingest_timestamp", "TIMESTAMP"),
                bigquery.SchemaField("source_path", "STRING"),
                bigquery.SchemaField("run_id", "STRING"),
            ],
            'channels': [
                bigquery.SchemaField("channel_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("channel_title", "STRING"),
                bigquery.SchemaField("channel_description", "STRING"),
                bigquery.SchemaField("country", "STRING"),
                bigquery.SchemaField("published_at", "TIMESTAMP"),
                bigquery.SchemaField("subscriber_count", "INTEGER"),
                bigquery.SchemaField("video_count", "INTEGER"),
                bigquery.SchemaField("view_count", "INTEGER"),
                bigquery.SchemaField("ingest_timestamp", "TIMESTAMP"),
                bigquery.SchemaField("source_path", "STRING"),
                bigquery.SchemaField("run_id", "STRING"),
            ],
            'comments': [
                bigquery.SchemaField("comment_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("video_id", "STRING"),
                bigquery.SchemaField("author_display_name", "STRING"),
                bigquery.SchemaField("text_display", "STRING"),
                bigquery.SchemaField("like_count", "INTEGER"),
                bigquery.SchemaField("published_at", "TIMESTAMP"),
                bigquery.SchemaField("ingest_timestamp", "TIMESTAMP"),
                bigquery.SchemaField("source_path", "STRING"),
                bigquery.SchemaField("run_id", "STRING"),
            ],
            'video_statistics': [
                bigquery.SchemaField("video_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("category_id", "STRING"),
                bigquery.SchemaField("tags", "STRING"),
                bigquery.SchemaField("duration", "STRING"),
                bigquery.SchemaField("view_count", "INTEGER"),
                bigquery.SchemaField("like_count", "INTEGER"),
                bigquery.SchemaField("comment_count", "INTEGER"),
                bigquery.SchemaField("favorite_count", "INTEGER"),
                bigquery.SchemaField("collected_at", "TIMESTAMP"),
                bigquery.SchemaField("ingest_timestamp", "TIMESTAMP"),
                bigquery.SchemaField("source_path", "STRING"),
                bigquery.SchemaField("run_id", "STRING"),
            ],
            'categories': [
                bigquery.SchemaField("category_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("category_title", "STRING"),
                bigquery.SchemaField("assignable", "BOOLEAN"),
                bigquery.SchemaField("region", "STRING"),
                bigquery.SchemaField("ingest_timestamp", "TIMESTAMP"),
                bigquery.SchemaField("source_path", "STRING"),
                bigquery.SchemaField("run_id", "STRING"),
            ]
        }
        
        # Create or recreate tables
        tables_info = []
        for table_name, schema in tables_config.items():
            table_ref = f"{project_id}.{dataset_id}.{table_name}"
            
            # Drop if requested
            if drop_existing:
                try:
                    client.delete_table(table_ref)
                    print(f"Dropped table {table_name}")
                except Exception:
                    pass
            
            # Create table
            table = bigquery.Table(table_ref, schema=schema)
            try:
                table = client.create_table(table, exists_ok=True)
                print(f"Table {table_name} ready")
                
                # Get row count
                query = f"SELECT COUNT(*) as cnt FROM `{table_ref}`"
                result = client.query(query).result()
                row_count = next(result).cnt
                
                tables_info.append({
                    "table": table_name,
                    "row_count": row_count
                })
            except Exception as e:
                print(f"Error creating table {table_name}: {e}")
                tables_info.append({
                    "table": table_name,
                    "error": str(e)
                })
        
        print("Schema setup complete")
        
        return {
            "status": "success",
            "message": "Schema created/verified successfully",
            "project": project_id,
            "dataset": dataset_id,
            "tables": tables_info,
            "drop_existing": drop_existing
        }, 200
        
    except Exception as e:
        error_msg = f"Schema setup failed: {str(e)}"
        print(f"{error_msg}")
        import traceback
        traceback.print_exc()
        
        return {
            "status": "error",
            "error": str(e)
        }, 500