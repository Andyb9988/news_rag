import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import json
import tiktoken
from utils import Helper
from google.cloud import storage
import json
from io import BytesIO
config_path = "config.json"

class YouTubeSearcher:
    def __init__(self, api_key):
        self.youtube = build('youtube', 'v3', developerKey=api_key)
    
    def get_video_list_by_search(self, max_results=5):
        """Searches for videos on YouTube using the Youtube Data API.

        Args:
            max_results (int, optional): The maximum number of results to return. Defaults to 15.

        Returns:
            list: A list of video URLs.
        """
        video_urls = []

        while True:
            query = str(input("What topic would you like summarised with Youtube videos: "))

            if query.lower() == 'done':
                break

            request = self.youtube.search().list(
                q=query,
                part='snippet',
                type='video',
                maxResults=max_results
            )
            response = request.execute()
            for item in response['items']:
                video_id = item['id']['videoId']
                video_url = f'https://www.youtube.com/watch?v={video_id}'
                video_urls.append(video_url)

            print(f"Number of items in the list: {len(video_urls)}")

            if len(video_urls) >= 5:
                print("Max video limit (5) reached")
                break

        return video_urls

    def get_video_metadata(self, video_urls):
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
                print("Video metadata found, now appending to list")
            else:
                print(f'No video found with ID {video_id}')
        
        #video_meta_df = pd.DataFrame(metadata_list)
        return metadata_list

    def upload_transcript_to_gcs(self, video_urls, bucket_name, folder_name):
        """
        Fetches transcripts of YouTube videos and uploads them directly to GCS as .txt files.

        Parameters:
        - video_urls (list): A list of YouTube video URLs.
        - bucket_name (str): The name of the GCS bucket.
        - folder_name (str): The name of the folder within the GCS bucket.
        """
        # Initialize the Google Cloud Storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        for url in video_urls:
            video_id = url.split('v=')[-1].split('&')[0]
            
            try:
                blob_path = f"{folder_name}/{video_id}.txt"
                blob = bucket.blob(blob_path)
                
                if not blob.exists():
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    # Open the blob for writing as a text file
                    with blob.open("w", encoding='utf-8') as f:
                        for sentence in transcript:
                            f.write(sentence['text'] + '\n')
                    print(f"Uploaded transcript for video {video_id} to {bucket_name}/{blob_path}")
                    
            except Exception as e:
                print(f"Error fetching/uploading transcript for {url}: {e}")

    def upload_dicts_to_gcs(self, bucket_name, folder_name, data_list):
        """
        Uploads a list of dictionaries as individual JSON files to GCS.

        Args:
            bucket_name (str): The name of the GCS bucket.
            folder_name (str): The name of the folder within the bucket.
            data_list (list): A list of dictionaries to upload.
        """    
    # Initialize the GCS client and get the bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Iterate through each dictionary in the list
        for data in data_list:
            video_id = data.get('Video_id')
            if not video_id:
                print("Dictionary missing 'video_id'. Skipping...")
                continue
            
            # Convert the dictionary to a JSON string
            data_str = json.dumps(data)
            
            # Create a blob (file) in the specified folder, named after the video ID
            blob_name = f"metadata/{video_id}.json"
            blob = bucket.blob(f"{folder_name}/{video_id}.json")
            
            #check if the blob exists
            if blob.exists():
                print(f"File {blob_name} already exists in bucket {bucket_name}")
            else:
        # Upload the JSON string to the blob if it does not exist
                blob.upload_from_string(data_str)
                print(f"Uploaded {video_id}.json to {folder_name}/ in bucket {bucket_name}")
     

class TranscriptProcessor:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    @staticmethod
    def lambda_clean_transcript_text(text):
        text = text.replace("[Music]", "")
        text = text.replace("\n", " ")
        text = text.replace("[Applause]", "")
        return text
    
    def create_transcript_column(self, transcripts_folder: str):
    # Loop through transcript files in the folder
        for filename in os.listdir(transcripts_folder):
            if filename.endswith('.txt'):
                video_id = os.path.splitext(filename)[0]  # Extract video ID from filename
                file_path = os.path.join(transcripts_folder, filename)

                with open(file_path, 'r', encoding='utf-8') as file:
                    transcript_content = file.read()
            # Find the corresponding row in the DataFrame and add the transcript content
                    self.df.loc[self.df['Video_id'] == video_id, 'transcript'] = transcript_content

        return self.df

    # Apply the function to the "text" column using lambda function
    def clean_transcript_text(self):
        self.df["clean_transcript"] = self.df["transcript"].apply(lambda x: self.lambda_clean_transcript_text(x))
        self.df.drop(columns=["transcript"], inplace=True)
        return self.df


    def count_number_of_tokens(self):
        token_counts = []
        for transcript in self.df['clean_transcript']:
            num_tokens = self.num_tokens_from_string(transcript, "cl100k_base")
            token_counts.append(num_tokens)
        self.df['num_tokens'] = token_counts
        return self.df

    def drop_columns_exceeding_token_length(self):
        threshold_tokens = 8192
        self.df = self.df[self.df['num_tokens'] <= threshold_tokens].reset_index(drop=True)
        print(f"The df has {self.df.shape[0]} rows")
        return self.df

def main():
    # Example usage
    helper = Helper(config_path)
    config = helper.load_config()
    youtube_api_key = config["YOUTUBE_API_KEY"]
    youtube_searcher = YouTubeSearcher(youtube_api_key)
    bucket_name = "youtube-fpl_data"
    meta_folder_name = "metadata"
    transcript_folder_name = "transcripts"

    video_urls = youtube_searcher.get_video_list_by_search(max_results=5)

    meta_data_list = youtube_searcher.get_video_metadata(video_urls)
    youtube_searcher.upload_dicts_to_gcs(bucket_name, meta_folder_name, meta_data_list)
    youtube_searcher.upload_transcript_to_gcs(video_urls, bucket_name, transcript_folder_name)
    #youtube_searcher.upload_list_to_gcs( bucket_name, transcript_folder_name, transcript_list)


    # Initialize TranscriptProcessor instance with your DataFrame
    #transcript_processor = TranscriptProcessor(df)

    # Call each method sequentially
   # transcript_processor.create_transcript_column('youtube_transcripts')
   #  transcript_processor.clean_transcript_text()
#  transcript_processor.count_number_of_tokens()
  #   transcript_processor.drop_columns_exceeding_token_length()
   # helper.save_df_to_csv(df)

if __name__ == '__main__':
    main()