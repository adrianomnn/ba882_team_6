import json
from typing import Dict, Optional, Any, List

import pandas as pd
import streamlit as st
from google.cloud import bigquery
from google.api_core.exceptions import NotFound, Forbidden, BadRequest
from google.oauth2 import service_account


def _info_from_exception(e: Exception) -> str:
    msg = str(e)
    if isinstance(e, NotFound):
        return "One or more tables not found. Please ensure required tables exist."
    if isinstance(e, Forbidden):
        return "Permission denied for BigQuery resources. Check IAM permissions."
    if isinstance(e, BadRequest):
        return "Invalid query. Check SQL and parameters."
    return msg


def get_bq_client() -> bigquery.Client:
    try:
        if hasattr(st, "secrets") and "gcp_service_account" in st.secrets and st.secrets.get("gcp_service_account"):
            sa_raw = st.secrets["gcp_service_account"]
            if isinstance(sa_raw, str):
                info = json.loads(sa_raw)
            else:
                info = dict(sa_raw)
            creds = service_account.Credentials.from_service_account_info(info)
            project_id = st.secrets.get("gcp_project_id")
            return bigquery.Client(credentials=creds, project=project_id)
        # Fall back to ADC
        return bigquery.Client()
    except Exception as e:
        st.info(f"Failed to initialize BigQuery client: {_info_from_exception(e)}")
        raise


def _to_bq_parameters(params: Dict[str, Any]) -> List[bigquery.ScalarQueryParameter]:
    bq_params: List[bigquery.ScalarQueryParameter] = []
    for name, value in params.items():
        if isinstance(value, bool):
            typ = "BOOL"
        elif isinstance(value, int):
            typ = "INT64"
        elif isinstance(value, float):
            typ = "FLOAT64"
        else:
            typ = "STRING"
        bq_params.append(bigquery.ScalarQueryParameter(name, typ, value))
    return bq_params


@st.cache_data(ttl=600, show_spinner=False)
def run_query(sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    try:
        client = get_bq_client()
        job_config = None
        if params:
            job_config = bigquery.QueryJobConfig(query_parameters=_to_bq_parameters(params))
        query_job = client.query(sql, job_config=job_config)
        df = query_job.result().to_dataframe(create_bqstorage_client=False)
        return df
    except Exception as e:
        st.info(f"BigQuery query failed: {_info_from_exception(e)}")
        return pd.DataFrame()
