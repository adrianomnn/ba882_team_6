import streamlit as st
import pandas as pd
from datetime import date
from utils.bq import run_query
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import altair as alt
from groq import Groq

# ---------------- App Page Title -----------------
st.set_page_config(page_title="Weekley Topic Summary", layout="wide")
st.title("Weekley Topic Summary")

# -------------------------------------------
# CONFIG
# -------------------------------------------
PROJECT = st.secrets.get("gcp_project_id") or st.secrets["gcp_service_account"]["project_id"]
DATASET = st.secrets.get("bq_dataset", "youtube_staging")
client = Groq(api_key=st.secrets["gcp_service_account"]["groq_api_key"])

st.markdown(
    """
    Select a week to retrieve all comments collected that week.
    """
)

@st.cache_data(ttl=3600)
def load_available_weeks():
    """
    Returns a DataFrame with columns: year, week, iso_week_label, week_start, week_end
    """
    sql = f"""
    SELECT
      EXTRACT(YEAR FROM DATE(published_at)) AS year,
      EXTRACT(ISOWEEK FROM DATE(published_at)) AS week,
      MIN(DATE(published_at)) AS week_start,
      MAX(DATE(published_at)) AS week_end
    FROM `{PROJECT}.{DATASET}.fact_comments` f
    JOIN `{PROJECT}.{DATASET}.dim_comments` d USING (comment_id)
    WHERE published_at IS NOT NULL
    GROUP BY year, week
    ORDER BY year DESC, week DESC
    """
    df = run_query(sql)
    if df.empty:
        return pd.DataFrame(columns=["year","week","iso_week_label","week_start","week_end"])
    df["week"] = df["week"].astype(int)
    df["year"] = df["year"].astype(int)
    df["iso_week_label"] = df.apply(
        lambda r: f"{r['year']}-W{str(r['week']).zfill(2)} ({r['week_start']} to {r['week_end']})",
        axis=1
    )
    return df

@st.cache_data(ttl=1800)
def load_comments_for_week(year:int, week:int, limit:int = None):
    """
    Returns comments for the given year and iso-week.
    """
    sql = f"""
    SELECT
      f.comment_id,
      d.comment_text,
      f.published_at,
      EXTRACT(YEAR FROM DATE(f.published_at)) as year_extracted,
      EXTRACT(ISOWEEK FROM DATE(f.published_at)) as week_extracted
    FROM `{PROJECT}.{DATASET}.fact_comments` f
    JOIN `{PROJECT}.{DATASET}.dim_comments` d
      ON f.comment_id = d.comment_id
    WHERE EXTRACT(YEAR FROM DATE(f.published_at)) = {year}
    AND EXTRACT(ISOWEEK FROM DATE(f.published_at)) = {week}
    ORDER BY f.published_at
    """
    df = run_query(sql)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"])
    if limit:
        return df.head(limit)
    return df


# --------------------- LLM Topic Summary ---------------------
def summarize_topics(comments, model="openai/gpt-oss-20b"):
    text = "\n".join(comments[:2000])

    prompt = f"""
    Summarize the main themes and discussion topics from the following YouTube comments.
    
    Structure output into:
    - Main Topics
    - Positive Themes
    - Negative Themes
    - Suggestions / Requests
    - Notable Observations

    COMMENTS:
    {text}
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# --------------------- UI: Week selector ---------------------
weeks_df = load_available_weeks()

if weeks_df.empty:
    st.info("No comments found in BigQuery (fact_comments / dim_comments) yet.")
    st.stop()

# Build options and mapping
options = weeks_df["iso_week_label"].tolist()
default_idx = 0
selected_label = st.selectbox("Select week", options, index=default_idx)

# Map back to numeric year/week
sel_row = weeks_df[weeks_df["iso_week_label"] == selected_label].iloc[0]
selected_year = int(sel_row["year"])
selected_week = int(sel_row["week"])

st.markdown(f"**Selected:** {selected_label}")

# --------------------- Load comments for selection ---------------------
limit = st.number_input("Max comments to preview (0 = all)", min_value=0, step=50, value=200)
limit_arg = None if limit == 0 else int(limit)

with st.spinner("Loading comments for selected week..."):
    comments_df = load_comments_for_week(selected_year, selected_week, limit=limit_arg)

st.subheader("🧵 Weekly Topic Summary")

if st.button("Generate Weekly Summary"):
    with st.spinner("Generating summary..."):
        summary_text = summarize_topics(comments_df["comment_text"].tolist())
    st.write(summary_text)