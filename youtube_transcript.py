import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import json
import tiktoken
config_path = "config.json"


try:
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
        youtube_api_key = config["YOUTUBE_API_KEY"]
except FileNotFoundError:
    print(f"Error: Configuration file '{config_path}' not found.")
    # Handle the case where the file is missing (optional)
    exit(1)

def youtube_search(max_results=15):
    query = str(input("What topic would you like summarised with Youtube videos: "))
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)

    request = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=max_results
    )
    response = request.execute()

    video_urls = []
    for item in response['items']:
        video_id = item['id']['videoId']
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        video_urls.append(video_url)

    return video_urls

def get_video_metadata(video_urls):
    metadata_list = []
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    for video_url in video_urls:
        video_id = video_url.split('v=')[-1].split('&')[0]

        request = youtube.videos().list(
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
        else:
            print(f'No video found with ID {video_id}')
    
    video_meta_df = pd.DataFrame(metadata_list)

    return video_meta_df

"""If subtitles are una"""

def save_youtube_transcripts(urls):
    # Create a folder if it doesn't exist
    folder_name = 'youtube_transcripts'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for url in urls:
        video_id = url.split('v=')[-1].split('&')[0]

        transcript_file_path = os.path.join(folder_name, f'{video_id}.txt')
        if os.path.exists(transcript_file_path):
            print(f"Transcript already exists for video {video_id}")
            continue
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            # Write transcript to a file
            with open(f'{folder_name}/{video_id}.txt', 'w', encoding='utf-8') as file:
                for sentence in transcript:
                    file.write(sentence['text'] + '\n')
            print(f"Transcript saved for video {video_id}")
        except Exception as e:
            print(f"Error fetching transcript for {url}: {e}")

def create_transcript_column(df):
  # Define the folder containing transcripts
    transcripts_folder = 'youtube_transcripts'

  # Loop through transcript files in the folder
    for filename in os.listdir(transcripts_folder):
        if filename.endswith('.txt'):
            video_id = os.path.splitext(filename)[0]  # Extract video ID from filename
            file_path = os.path.join(transcripts_folder, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                transcript_content = file.read()
          # Find the corresponding row in the DataFrame and add the transcript content
                df.loc[df['Video_id'] == video_id, 'transcript'] = transcript_content

    return df

def lambda_clean_transcript_text(text):
    text = text.replace("[Music]", "")
    text = text.replace("\n", " ")
    text = text.replace("[Applause]", "")
    return text

# Apply the function to the "text" column using lambda function
def clean_transcript_text(df):
    df["clean_transcript"] = df["transcript"].apply(lambda x: lambda_clean_transcript_text(x))
    df.drop(columns=["transcript"], inplace=True)
    return df

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Assuming 'df' is your DataFrame containing the transcript column
def count_tokens(df):
    token_counts = []
    for transcript in df['clean_transcript']:
        num_tokens = num_tokens_from_string(transcript, "cl100k_base")
        token_counts.append(num_tokens)
    df['num_tokens'] = token_counts
    return df

def drop_columns_exceeding_token(df):
# Define the threshold for the number of tokens
    threshold_tokens = 8192

    # Iterate through the dataset and drop rows where the number of tokens is less than 8192
    for index, row in df.iterrows():
        if row['num_tokens'] > threshold_tokens:
            df.drop(index, inplace=True)

    # Reset the index after dropping rows
    df.reset_index(drop=True, inplace=True)
    print(f"The df has {df.shape[0]} rows")
    return df


def save_df_to_csv(dataframe):
    """
    Save a Pandas DataFrame to a CSV file.

    Parameters:
    dataframe (pandas.DataFrame): The DataFrame to be saved.
    file_path (str): The file path where the CSV file will be saved.

    Returns:
    None
    """

    folder_name = 'youtube_dataframe'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
     # Save DataFrame to CSV
    dataframe.to_csv(f"{folder_name}/output.csv", index=False)
    print(f"DataFrame successfully saved to {folder_name}/output.csv")


def main():
    # Example usage
    video_urls = youtube_search(max_results=15)
    df =  get_video_metadata(video_urls)
    save_youtube_transcripts(video_urls)
    df = create_transcript_column(df)
    df = clean_transcript_text(df)
    df = count_tokens(df)
    df = drop_columns_exceeding_token(df)
    save_df_to_csv(df)

if __name__ == '__main__':
    main()