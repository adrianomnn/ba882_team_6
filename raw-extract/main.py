# extract_youtube.py
import functions_framework
from google.cloud import storage
import datetime, uuid, json
from youtube_api import get_video, get_channel_details, get_video_statistics, get_video_comments, get_video_categories

project_id = 'adrineto-qst882-fall25'
bucket_name = 'adrineto-ba882-fall25-team-6'

def upload_to_gcs(bucket_name, path, run_id, data):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = f"{path}/{run_id}/data.json"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data)
    print(f"Uploaded {blob_name} to {bucket_name}")
    return {'bucket_name': bucket_name, 'blob_name': blob_name}

@functions_framework.http
def task(request):
    query = request.args.get("query", "Data Engineering")
    run_id = uuid.uuid4().hex[:12]
    print(f"Query: {query}, Run ID: {run_id}")

    # Extract from YouTube
    videos_df = get_video(query, max_results=50)
    channel_ids = videos_df["channel_id"].dropna().unique().tolist()
    channels_df = get_channel_details(channel_ids)
    stats_df = get_video_statistics(videos_df["video_id"].tolist())
    if not videos_df.empty:
        comments_df = get_video_comments(videos_df["video_id"].iloc[0])
    else:
        comments_df = None

    categories_df = get_video_categories(region_code="US")

    data = {
        "query": query,
        "videos": videos_df.to_dict(orient="records"),
        "channels": channels_df.to_dict(orient="records"),
        "video_stats": stats_df.to_dict(orient="records"),
        "comments": comments_df.to_dict(orient="records") if comments_df is not None else [],
        "categories": categories_df.to_dict(orient="records"),
        "extracted_at": datetime.datetime.utcnow().isoformat()
    }

    json_str = json.dumps(data, default=str)
    date_path = datetime.datetime.utcnow().strftime("%Y%m%d")
    gcs_path = upload_to_gcs(bucket_name, f"raw/youtube/query={query}/date={date_path}", run_id, json_str)

    return {"run_id": run_id, **gcs_path}, 200
