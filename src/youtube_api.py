import os
import requests
import pandas as pd

API_KEY = os.getenv("YOUTUBE_API_KEY")
BASE_URL = "https://www.googleapis.com/youtube/v3"

def get_channel_details(channel_ids):
    """
    Retrieves channel metadata for one or multiple YouTube channel IDs.
    Returns a DataFrame with channel details.
    """
    if isinstance(channel_ids, str):
        channel_ids = [channel_ids]
    elif not isinstance(channel_ids, list):
        raise ValueError("channel_ids must be a string or list of strings")

    channels_data = []

    for i in range(0, len(channel_ids), 50):  # YouTube allows max 50 IDs per call
        ids = ",".join(channel_ids[i:i+50])
        url = f"{BASE_URL}/channels?part=snippet,statistics&id={ids}&key={API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Skip if no valid channels found
        if "items" not in data or not data["items"]:
            print(f"No valid data returned for batch: {channel_ids[i:i+50]}")
            continue

        for item in data["items"]:
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})

            channels_data.append({
                "channel_id": item.get("id"),
                "channel_title": snippet.get("title"),
                "description": snippet.get("description"),
                "published_at": snippet.get("publishedAt"),
                "country": snippet.get("country"),
                "view_count": int(stats.get("viewCount", 0)),
                "subscriber_count": int(stats.get("subscriberCount", 0)),
                "video_count": int(stats.get("videoCount", 0))
            })

    return pd.DataFrame(channels_data)


def get_video(query, max_results=20):
    """
    Retrieves YouTube videos for a given search query.
    Returns only basic info: video_id, channel_id, title, description, and published_at.
    """
    search_url = f"{BASE_URL}/search?part=snippet&q={query}&type=video&maxResults={max_results}&key={API_KEY}"
    response = requests.get(search_url)
    response.raise_for_status()
    items = response.json().get("items", [])

    videos = []
    for item in items:
        # Safely extract video ID (some items may not have it)
        video_id = item.get("id", {}).get("videoId")
        if not video_id:
            continue  # skip bad or missing items

        snippet = item.get("snippet", {})
        videos.append({
            "video_id": video_id,
            "channel_id": snippet.get("channelId"),
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "published_at": snippet.get("publishedAt"),
            "search_query": query
        })

    return pd.DataFrame(videos)

def get_video_statistics(video_ids: list):
    """
    Retrieves detailed statistics and metadata (duration, category, tags) for a list of video IDs.
    Returns view_count, like_count, favorite_count, comment_count, duration, category_id, and tags.
    """
    if not video_ids:
        return pd.DataFrame()

    stats_data = []
    for i in range(0, len(video_ids), 50):  # API allows up to 50 IDs per request
        ids = ",".join(video_ids[i:i+50])
        url = f"{BASE_URL}/videos?part=snippet,contentDetails,statistics&id={ids}&key={API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        for item in response.json().get("items", []):
            snippet = item.get("snippet", {})
            content = item.get("contentDetails", {})
            stats = item.get("statistics", {})

            stats_data.append({
                "video_id": item["id"],
                "category_id": snippet.get("categoryId"),
                "tags": ",".join(snippet.get("tags", [])) if snippet.get("tags") else None,
                "duration": content.get("duration"),
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "favorite_count": int(stats.get("favoriteCount", 0)),
                "comment_count": int(stats.get("commentCount", 0))
            })

    return pd.DataFrame(stats_data)

def get_video_comments(video_id: str, max_results=100):
    comments = []
    next_page = None

    while True:
        url = f"{BASE_URL}/commentThreads?part=snippet&videoId={video_id}&maxResults={max_results}&key={API_KEY}"
        if next_page:
            url += f"&pageToken={next_page}"

        resp = requests.get(url).json()
        for item in resp.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["id"],
                "video_id": video_id,
                "author_display_name": snippet.get("authorDisplayName"),
                "text_display": snippet.get("textDisplay"),
                "like_count": int(snippet.get("likeCount", 0)),
                "published_at": snippet.get("publishedAt")
            })

        next_page = resp.get("nextPageToken")
        if not next_page:
            break

    return pd.DataFrame(comments)

def get_video_categories(region_code="US"):
    url = f"{BASE_URL}/videoCategories?part=snippet&regionCode={region_code}&key={API_KEY}"
    response = requests.get(url)
    response.raise_for_status()

    categories = []
    for item in response.json().get("items", []):
        snippet = item["snippet"]
        categories.append({
            "category_id": item["id"],
            "title": snippet["title"],
            "assignable": snippet["assignable"]
        })

    return pd.DataFrame(categories)
