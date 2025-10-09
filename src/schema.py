from google.cloud import bigquery

def create_youtube_tables(project_id: str, dataset_id: str):
    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_id)

    tables = {
        "channels": [
            bigquery.SchemaField("channel_id", "STRING", "REQUIRED"),
            bigquery.SchemaField("channel_title", "STRING"),
            bigquery.SchemaField("description", "STRING"),
            bigquery.SchemaField("published_at", "TIMESTAMP"),
            bigquery.SchemaField("country", "STRING"),
            bigquery.SchemaField("view_count", "INTEGER"),
            bigquery.SchemaField("subscriber_count", "INTEGER"),
            bigquery.SchemaField("video_count", "INTEGER"),
        ],
        "videos": [
            bigquery.SchemaField("video_id", "STRING", "REQUIRED"),
            bigquery.SchemaField("channel_id", "STRING"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("description", "STRING"),
            bigquery.SchemaField("published_at", "TIMESTAMP"),
        ],
        "video_statistics": [
            bigquery.SchemaField("video_id", "STRING", "REQUIRED"),
            bigquery.SchemaField("view_count", "INTEGER"),
            bigquery.SchemaField("like_count", "INTEGER"),
            bigquery.SchemaField("duration", "STRING"),
            bigquery.SchemaField("favorite_count", "INTEGER"),
            bigquery.SchemaField("comment_count", "INTEGER"),
            bigquery.SchemaField("category_id", "STRING"),
            bigquery.SchemaField("tags", "STRING"),
        ],
        "comments": [
            bigquery.SchemaField("comment_id", "STRING", "REQUIRED"),
            bigquery.SchemaField("video_id", "STRING"),
            bigquery.SchemaField("author_display_name", "STRING"),
            bigquery.SchemaField("text_display", "STRING"),
            bigquery.SchemaField("like_count", "INTEGER"),
            bigquery.SchemaField("published_at", "TIMESTAMP"),
        ],
        "video_categories": [
            bigquery.SchemaField("category_id", "STRING", "REQUIRED"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("assignable", "BOOLEAN"),
        ],
    }

    for table_name, schema in tables.items():
        table_ref = dataset_ref.table(table_name)
        table = bigquery.Table(table_ref, schema=schema)
        try:
            client.create_table(table)
            print(f"Created table: {dataset_id}.{table_name}")
        except Exception as e:
            if "Already Exists" in str(e):
                print(f"Table {dataset_id}.{table_name} already exists.")
            else:
                raise e
