import streamlit as st
from utils.bq import run_query

st.set_page_config(page_title="Sports Video Intelligence", layout="wide")

st.title("Sports Video Intelligence")
st.write("Phase 1 Data Ops Dashboard (BigQuery)")

with st.container():
    st.subheader("Health")
    project_id = st.secrets.get("gcp_project_id") if hasattr(st, "secrets") else None
    dataset = st.secrets.get("bq_dataset", "youtube_staging") if hasattr(st, "secrets") else "youtube_staging"

    cols = st.columns(2)
    with cols[0]:
        if project_id:
            st.metric(label="GCP Project ID", value=project_id)
        else:
            st.info("Set 'gcp_project_id' in Streamlit secrets to display project info.")
    with cols[1]:
        st.metric(label="BigQuery Dataset", value=dataset)

    smoke = run_query("SELECT 1 AS ok")
    if smoke is not None and not smoke.empty and "ok" in smoke.columns:
        st.success("BigQuery connection OK (SELECT 1)")
    else:
        st.info("Could not run smoke test. Ensure BigQuery access is configured and tables exist.")

st.divider()
st.write("Use the sidebar or the '📊 Dashboard' page for KPIs and charts. The app auto-discovers pages from the 'pages/' folder.")
