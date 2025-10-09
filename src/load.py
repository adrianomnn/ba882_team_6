from google.cloud import bigquery
import pandas as pd

def load_to_bigquery(df: pd.DataFrame, table_id: str, write_disposition="WRITE_APPEND"):
    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    print(f"Loaded {len(df)} rows into {table_id}")
