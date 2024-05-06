import pandas as pd
from typing import List, Dict, Union, Tuple
import json
from utils.helper import Helper
from google.cloud import storage
import json
from io import BytesIO
config_path = "utils/config.json"

from pinecone import ServerlessSpec, PodSpec
from pinecone import Pinecone as Pinecone_Client
from openai import OpenAI
import langchain
import json
import string
from langchain_community.document_loaders import GCSFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import logging

class Pinecone:
    def __init__(self, api_key: str):
        self.pc = Pinecone_Client(api_key = api_key)
        self.index_name = 'youtube-transcripts'

    def create_pinecone_index(self):
        use_serverless = True
        if self.index_name in self.pc.list_indexes().names():
            self.index = self.pc.Index(self.index_name)

        else:
        # create a new index
            if use_serverless:
                spec = ServerlessSpec(cloud='aws', region='us-west-2')
            else:
                # if not using a starter index, you should specify a pod_type too
                spec = PodSpec()
               
            self.pc.create_index(
                    self.index_name,
                    dimension=1536,  # dimensionality of text-embedding-ada-002
                    metric='cosine',
                    spec=spec
                )
            while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
            self.index = self.pc.Index(self.index_name)
            
        logging.info(self.index.describe_index_stats())
        return self.index
    
    def upsert_data(self, data_file: Union[Dict, List[Dict]]) -> None:
        logging.info(f"After adding the new documents {self.index.describe_index_stats()}")
        self.index.upsert(data_file)

class PrepareTextForVDB:
    def __init__(self):
        self.folder_path = "clean_transcripts"
        self.bucket_name = "youtube-fpl_data"
        self.project_name = "youtube-to-gpt"
        self.client = OpenAI()
    

    def list_blobs_in_folder(self) -> List[storage.Blob]:
        """Lists all the blobs in the specified GCS folder."""
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(self.bucket_name)
        
        blobs = bucket.list_blobs(prefix=self.folder_path) 
        logging.info("All blobs are loaded and added to a list.")   
        return list(blobs)

    def load_blobs(self, blobs: List[storage.Blob]) -> List:
        documents = []
        for blob in blobs:
            try:
                loader = GCSFileLoader(project_name=self.project_name, bucket=self.bucket_name, blob=blob.name)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                logging.info("Documents are loaded and have been added to a list")
            except PermissionError as e:
                logging.error(f"Permission denied for blob {blob.name}. Skipping download.")
                logging.error(f"Error details: {str(e)}")
        return documents

    def split_documents(self, documents, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        split_docs = text_splitter.split_documents(documents)
        logging.info("The list of documents are now recursively split.")
        return split_docs

    def _create_embedding_list(self, split_docs: List) -> List[List[float]]:
        embed_list = [self.client.embeddings.create(input=[i.page_content], model="text-embedding-3-small").data[0].embedding for i in split_docs]
        return embed_list
    
    def _create_metatdata_list(self, split_docs: List) -> List[dict]:
        meta_list = [{"video_id": i.metadata["source"], "text": i.page_content} for i in split_docs]
        return meta_list
    
    def _create_ids(self, split_docs: List) -> List[str]:
        ids = [str(i) for i in range(0, len(split_docs))]
        return ids
    
    def create_zip_file(self, split_docs: List) -> List[Tuple[str, List[float], dict]]:
        ids = self._create_ids(split_docs)
        embed_list = self._create_embedding_list(split_docs)
        meta_list = self._create_metatdata_list(split_docs)
        logging.info("Zip file create with ID, embeddings and metadata.")
        return zip(ids, embed_list, meta_list)
    
def main():
    helper = Helper(config_path)
    config = helper.load_config()
    pinecone_api = config["PINECONE_API_KEY"]
    openai_api = config["OPENAI_API_KEY"]

        #Create Pinecone Index
    pinecone = Pinecone(pinecone_api)
    index = pinecone.create_pinecone_index()
    #Prepare text for upsert
    prep_text_for_rag = PrepareTextForVDB()
    blob_list = prep_text_for_rag.list_blobs_in_folder()
    blob_docs = prep_text_for_rag.load_blobs(blob_list)
    split_documents = prep_text_for_rag.split_documents(blob_docs)
    zip_file = prep_text_for_rag.create_zip_file(split_documents)
    #Upsert transcript
    pinecone.upsert_data(zip_file)

if __name__ == '__main__':
    main()