import streamlit as st
import pandas as pd
from datetime import date
from utils.bq import run_query
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------- App Page Title -----------------
st.set_page_config(page_title="Weekley Comments Sentiment Analysis", layout="wide")
st.title("Weekley Comments Sentiment Analysis")

# -------------------------------------------
# CONFIG
# -------------------------------------------
PROJECT = st.secrets.get("gcp_project_id") or st.secrets["gcp_service_account"]["project_id"]
DATASET = st.secrets.get("bq_dataset", "youtube_staging")

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

# --------------------- Vader Sentiment Analysis ---------------------
def apply_vader(df):
    analyzer = SentimentIntensityAnalyzer()
    df["vader"] = df["comment"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    return df

def vader_label(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"

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

st.markdown("### Comments summary")
if comments_df.empty:
    st.warning("No comments for this week.")
else:
    c1, c2, c3 = st.columns([1, 1, 1.3])
    c1.metric("Comments (rows)", int(len(comments_df)))
    # proportion of comments with non-empty text
    non_empty = comments_df["comment_text"].notna().sum()
    c2.metric("Comments with text", int(non_empty))
    # first / last timestamps
    c3.metric("Date range", f"{comments_df['published_at'].min().date()} → {comments_df['published_at'].max().date()}")

    st.markdown("#### Sample comments")
    st.dataframe(comments_df.reset_index(drop=True).head(200))

    st.markdown("#### Quick text preview (first 10)")
    for idx, row in comments_df.head(10).iterrows():
        st.write(f"- `{row['comment_id']}` — {row['published_at']:%Y-%m-%d %H:%M} — {row['comment_text'][:400]}")

import streamlit as st
import pandas as pd
from datetime import date
from utils.bq import run_query
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------- App Page Title -----------------
st.set_page_config(page_title="Weekley Comments Sentiment Analysis", layout="wide")
st.title("Weekley Comments Sentiment Analysis")

# -------------------------------------------
# CONFIG
# -------------------------------------------
PROJECT = st.secrets.get("gcp_project_id") or st.secrets["gcp_service_account"]["project_id"]
DATASET = st.secrets.get("bq_dataset", "youtube_staging")

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

# --------------------- Vader Sentiment Analysis ---------------------
def apply_vader(df):
    analyzer = SentimentIntensityAnalyzer()
    df["vader"] = df["comment"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    return df

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

st.markdown("### Comments summary")
if comments_df.empty:
    st.warning("No comments for this week.")
else:
    c1, c2, c3 = st.columns([1, 1, 1.3])
    c1.metric("Comments (rows)", int(len(comments_df)))
    # proportion of comments with non-empty text
    non_empty = comments_df["comment_text"].notna().sum()
    c2.metric("Comments with text", int(non_empty))
    # first / last timestamps
    c3.metric("Date range", f"{comments_df['published_at'].min().date()} → {comments_df['published_at'].max().date()}")

    st.markdown("#### Sample comments")
    st.dataframe(comments_df.reset_index(drop=True).head(200))

    st.markdown("#### Quick text preview (first 10)")
    for idx, row in comments_df.head(10).iterrows():
        st.write(f"- `{row['comment_id']}` — {row['published_at']:%Y-%m-%d %H:%M} — {row['comment_text'][:400]}")

st.subheader("📘 VADER Sentiment Analysis")

if comments_df.empty:
    st.info("No comments found for selected week.")
else:
    # Work on a copy to avoid warnings
    df_week = comments_df.copy()

    # Ensure comment_text is string
    df_week["comment_text"] = df_week["comment_text"].fillna("").astype(str)

    with st.spinner("Running VADER sentiment analysis..."):
        analyzer = SentimentIntensityAnalyzer()
        df_week["vader"] = df_week["comment_text"].apply(
            lambda x: analyzer.polarity_scores(x)["compound"]
        )

        # Apply labels
        def vader_label(score):
            if score >= 0.05:
                return "positive"
            elif score <= -0.05:
                return "negative"
            return "neutral"

        df_week["vader_label"] = df_week["vader"].apply(vader_label)

    st.success("VADER sentiment calculated successfully!")

    # Display sample output
    st.markdown("### Sample Results")
    st.dataframe(df_week[["comment_id", "comment_text", "vader", "vader_label"]].head(20))

    # Summary aggregated metrics
    summary = (
        df_week["vader_label"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )

    st.markdown("### Sentiment Breakdown")
    st.bar_chart(summary.set_index("sentiment"))