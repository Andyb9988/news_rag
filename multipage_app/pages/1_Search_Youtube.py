import streamlit as st
from utils.yt_to_gcs import YouTubeSearcher
from utils.data_processor import TranscriptProcessor
from utils.vector_database import PineconeHelper, PrepareTextForVDB
from utils.helper import Helper
config_path = "multipage_app/utils/config.json"
helper = Helper(config_path)
config = helper.load_config()

youtube_api_key = config["YOUTUBE_API_KEY"]
pinecone_api = config["PINECONE_API_KEY"]
openai_api = config["OPENAI_API_KEY"]

yt_searcher = YouTubeSearcher(api_key=youtube_api_key)

def search_and_upload_video(folder_name, input_query):
        #Upload into GCS Bucket
        yt_searcher.proccess_videos_to_gcs(folder_name=folder_name, user_query=input_query)
        #Filter, Clean and process
        tp = TranscriptProcessor(folder_name=folder_name)
        tp.process_transcripts
        #Upload to pinecone
        pinecone = PineconeHelper(pinecone_api, index_name=folder_name)
        index = pinecone.pinecone_index()

        prep_text_for_rag = PrepareTextForVDB(folder_name=folder_name)
        blob_list = prep_text_for_rag.list_blobs_in_folder()
        blob_docs = prep_text_for_rag.load_blobs(blob_list)
        split_documents = prep_text_for_rag.split_documents(blob_docs)
        zip_file = prep_text_for_rag.create_zip_file(split_documents)
        #Upsert transcript
        pinecone.upsert_data(zip_file)


page_config = {
    "page_title": "Search Topics on Youtube.",
    "layout": "centered",
}
st.set_page_config(**page_config)
st.markdown("<h1 style='text-align: center;'> Ask Youtube Anything </h1>", unsafe_allow_html=True)

# Text input for search query
input_query = st.text_input('Enter a topic to search on YouTube:', '')

col1, col2, col3, col4 = st.columns(4)
with col1:
    fpl = st.checkbox('fpl')
with col2:
    golf = st.checkbox('golf')
with col3:
    xrp = st.checkbox('xrp')
with col4:
    other = st.checkbox('ther')

# Search button
# Horizontal checkboxes for topic selection
# 1. upload transcript and metadata into appropriate folders.
if st.button('Search'):
    if input_query:
        if fpl:
            st.write(f'Searching for: {input_query} in FPL database')
            # Add your database querying logic here for FPL
            folder_name = "fpl"
            search_and_upload_video(folder_name, input_query)
            
        if golf:
            st.write(f'Searching for: {input_query} in Golf database')
            folder_name = "golf"
            search_and_upload_video(folder_name, input_query)
            
        if xrp:
            st.write(f'Searching for: {input_query} in XRP database')
            # Add your database querying logic here for XRP
            folder_name = "xrp"
            search_and_upload_video(folder_name, input_query)

        if other:
            st.write(f'Searching for: {input_query} in Other database')
            # Add your database querying logic here for Other
            folder_name = "other"
            search_and_upload_video(folder_name, input_query)
    else:
        st.error('Please enter a search term.')