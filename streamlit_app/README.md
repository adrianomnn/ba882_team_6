# Sports Video Intelligence - Phase 1 (Data Ops)

Minimal Streamlit dashboard wired to BigQuery core tables.

## Run locally

```
cd streamlit_app
pip install -r requirements.txt
gcloud auth application-default login
streamlit run app.py
```

## Deploy on Streamlit Cloud

- Point to `streamlit_app/app.py` as the entry file.
- Set Secrets in the app settings with keys:
  - `gcp_project_id`
  - `bq_dataset`
  - `gcp_service_account` (paste JSON content)
