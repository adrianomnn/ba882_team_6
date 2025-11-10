import json
import collections.abc as cabc  # for Mapping / AttrDict check
from typing import Optional, Dict
import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# --- Optional debug expander (you can move this to app.py later) ---
with st.expander("Debug secrets", expanded=False):
    st.write("secrets keys:", list(st.secrets.keys()))
    sa = st.secrets.get("gcp_service_account")
    st.write("gcp_service_account type:", type(sa).__name__)
    st.write("gcp_project_id:", st.secrets.get("gcp_project_id"))
# -------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_bq_client() -> bigquery.Client:
    """
    Build a BigQuery client from Streamlit secrets (string JSON or Mapping),
    falling back to ADC if no secrets are present.
    """
    info = st.secrets.get("gcp_service_account", None)

    # If it's a JSON string inside TOML, parse it
    if isinstance(info, str):
        info = json.loads(info)

    # Streamlit may return a SecretDict / AttrDict (a Mapping). Coerce to plain dict.
    if isinstance(info, cabc.Mapping):
        info = dict(info)

    if isinstance(info, dict):
        creds = service_account.Credentials.from_service_account_info(info)
        project_id = st.secrets.get("gcp_project_id") or info.get("project_id")
        if not project_id:
            raise RuntimeError("No project_id found in secrets or service account JSON.")
        return bigquery.Client(project=project_id, credentials=creds)

    # Fallback: Application Default Credentials (env var or gcloud ADC)
    return bigquery.Client()

@st.cache_data(ttl=600)
def run_query(sql: str, params: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Run a query and return a DataFrame. Supports optional scalar string params.
    """
    try:
        client = get_bq_client()
        if params:
            from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter
            job_config = QueryJobConfig(
                query_parameters=[
                    ScalarQueryParameter(k, "STRING", v) for k, v in params.items()
                ]
            )
            job = client.query(sql, job_config=job_config)
        else:
            job = client.query(sql)
        return job.result().to_dataframe()
    except Exception as e:
        st.error(f"BigQuery query failed: {e}")
        return pd.DataFrame()
