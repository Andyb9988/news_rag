from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pandas as pd
from typing import List, Dict, Union, Tuple
from youtube_transcript_api import YouTubeTranscriptApi
import json
#from utils.helper import Helper
from utils.helper import Helper
from google.cloud import storage
import json
from io import BytesIO
import logging
config_path = "utils/config.json"
import logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

class YouTubeSearcher:
    def __init__(self, api_key):
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.bucket_name = "youtube-transcript-data"
    
    def get_video_list_by_search(self, user_query: str, max_results: int = 2) -> List[str]:
        """Searches for videos on YouTube using the Youtube Data API.

        Args:
            max_results (int, optional): The maximum number of results to return per search. Defaults to 5.

        Returns:
            list: A list of video URLs.
        """
        video_urls = []
        while True:
            query = user_query
            #str(input("What topic would you like summarised with Youtube videos (type 'done' to finish): "))

            if query.lower() == 'done':
                break

            request = self.youtube.search().list(
                q=query,
                part='snippet',
                type='video',
                maxResults=max_results
            )
            response = request.execute()

            new_video_urls = [f'https://www.youtube.com/watch?v={item["id"]["videoId"]}' for item in response['items']]
            
            video_urls.extend(new_video_urls)
            logging.info(f"The number of items added to the list: {len(video_urls)}", video_urls)
            
            if len(video_urls) >= 60:
                logging.info("Max video limit (60) reached")
                break

        return video_urls

    def get_video_metadata(self, video_urls: List[str]) -> List[Dict]:
        """
        Fetches metadata for a list of YouTube videos.

        This function iterates over a list of YouTube video URLs, extracts the video ID from each URL,
        and uses the YouTube Data API to retrieve the video's metadata.
        The metadata is then compiled into a DataFrame for easy manipulation.

        Parameters:
        - video_urls (list): A list of YouTube video URLs.

        Returns:
        - List of dictionaries: A list containing the metadata for each video.
        """
        metadata_list = []
        for video_url in video_urls:
            video_id = video_url.split('v=')[-1].split('&')[0]
            try:

                request = self.youtube.videos().list(
                    part='snippet',
                    id=video_id
                )
                response = request.execute()
                video_items = response['items']

                if video_items:
                    video_item = video_items[0]
                    snippet = video_item['snippet']
                    metadata = {
                        'Title': snippet['title'],
                        'Published_at': snippet['publishedAt'],
                        'Channel': snippet["channelTitle"],
                        'URL': f"https://www.youtube.com/watch?v={video_id}",
                        "Video_id": video_id
                    }
                    metadata_list.append(metadata)
                    logging.info("metadata added to a list successfully")
                else:
                    logging.warning(f'No video found with ID {video_id}')
            
            except Exception as e:
                logging.error(f"Error fetching video metadata for ID {video_id}: {e}")


        return metadata_list

    def upload_transcript_to_gcs(self, video_urls: List[str], folder_name: str):
        """
        Fetches transcripts of YouTube videos and uploads them directly to GCS as .txt files.

        Parameters:
        - video_urls (list): A list of YouTube video URLs.
        - bucket_name (str): The name of the GCS bucket.
        - folder_name (str): The name of the folder within the GCS bucket.
        """
        # Initialize the Google Cloud Storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        
        # iterate through urls and split video id.
        for url in video_urls:
            video_id = url.split('v=')[-1].split('&')[0]
            
            try:
                blob_path = f"{folder_name}/{folder_name}_transcripts/{video_id}.txt"
                blob = bucket.blob(blob_path)
                
                # if the blob does not already exist, get transcript.
                if not blob.exists():
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    # Open the blob for writing as a text file.
                    with blob.open("w", encoding='utf-8') as f:
                        for sentence in transcript:
                            f.write(sentence['text'] + '\n')
                    logging.info(f"Uploaded transcript for video {video_id} to {self.bucket_name}/{blob_path}")
                    
            except Exception as e:
                logging.error(f"Error fetching/uploading transcript for {url}: {e}")

    def upload_dicts_to_gcs(self, folder_name: str, data_list: List[Dict]):
        """
        Uploads a list of dictionaries as individual JSON files to GCS.

        Args:
            bucket_name (str): The name of the GCS bucket.
            folder_name (str): The name of the folder within the bucket.
            data_list (list): A list of dictionaries to upload.
        """    
        # Initialize the GCS client and get the bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        
        # Iterate through each dictionary in the list
        for data in data_list:
            video_id = data.get('Video_id')
            if not video_id:
                logging.info(f"Dictionary missing {video_id}. Skipping...")
                continue
            
            # Convert the dictionary to a JSON string
            data_str = json.dumps(data)
            
            try:
                blob_path = f"{folder_name}/{folder_name}_metadata/{video_id}.json"
                blob = bucket.blob(blob_path)
                
                #check if the blob exists
                if blob.exists():
                    logging.info(f"File {blob} already exists in bucket {self.bucket_name}")
                else:
                # Upload the JSON string to the blob if it does not exist
                    blob.upload_from_string(data_str)
                    logging.info(f"Uploaded {video_id}.json to {folder_name}/ in bucket {self.bucket_name}")

            except Exception as e:
                logging.error(f"Error fetching/uploading metadat for {video_id}: {e}")


    def proccess_videos_to_gcs(self, folder_name, user_query):
        #folder_name = "golf"
            # Search for the youtube URL's
        video_urls = self.get_video_list_by_search(user_query, max_results=2)

        # Gets video metadata and uploads metadata and transcript to a gcs bucket
        meta_data_list = self.get_video_metadata(video_urls)
        self.upload_dicts_to_gcs(folder_name, meta_data_list)
        self.upload_transcript_to_gcs(video_urls, folder_name)