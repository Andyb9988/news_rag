import pandas as pd
import os
from openai import OpenAI
import math
import regex as re
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import openai 
import langchain
import json
from utils import Helper
config_path = "config.json"

df = pd.read_csv(r"youtube_dataframe/output.csv")

pc = Pinecone(
        api_key=pinecone_api
    )
def create_pinecone_index():
    index_name = 'fpl-rag'
    # Now do stuff
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-west-2'
            )
        )
    index = pc.Index(index_name)
    return index


def main():
    helper = Helper(config_path)
    config = helper.load_config()
    youtube_api_key = config["YOUTUBE_API_KEY"]
    hf_token = config["HF_TOKEN"]
    pinecone_api = config["PINECONE_API_KEY"]
    openai_api = config["OPENAI_API_KEY"]
    create_pinecone_index()
   

if __name__ == "__main__":
    main()