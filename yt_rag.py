import regex as re
from pinecone import ServerlessSpec, PodSpec
from pinecone import Pinecone as Pinecone_Client
import openai 
from openai import OpenAI
import langchain
import json
import string
import re
from utils import Helper
from google.cloud import storage
from langchain_community.document_loaders import GCSFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore  
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import time

config_path = "config.json"


class Pinecone:
    def __init__(self, api_key):
        self.pc = Pinecone_Client(api_key = api_key)
        self.index_name = 'youtube-transcripts'

    def create_pinecone_index(self):
        use_serverless = True
        index_name = self.index_name
        if index_name in self.pc.list_indexes().names():
            self.index = self.pc.Index(self.index_name)

        else:
    # create a new index
            if use_serverless:
                spec = ServerlessSpec(cloud='aws', region='us-west-2')
            else:
                # if not using a starter index, you should specify a pod_type too
                spec = PodSpec()
            
            self.pc.create_index(
                index_name,
                dimension=1536,  # dimensionality of text-embedding-ada-002
                metric='cosine',
                spec=spec
            )
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            self.index = self.pc.Index(self.index_name)
        
        print(self.index.describe_index_stats())
        return self.index
    
    def upsert_data(self, data_file):
        self.index.upsert(data_file)

class PrepareTextForRag:
    def __init__(self):
        self.folder_path = "clean_transcripts"
        self.bucket_name = "youtube-fpl_data"
        self.project_name = "youtube-to-gpt"
        self.client = OpenAI()
    

    def list_blobs_in_folder(self):
        """Lists all the blobs in the specified GCS folder."""
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(self.bucket_name)
        
        blobs = bucket.list_blobs(prefix=self.folder_path) 
        print("All blobs are loaded")   
        return list(blobs)

    def load_blobs(self, blobs):
        documents = []
        for blob in blobs:
            try:
                loader = GCSFileLoader(project_name=self.project_name, bucket=self.bucket_name, blob=blob.name)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                print("Docs created")
            except PermissionError as e:
                print(f"Permission denied for blob {blob.name}. Skipping download.")
                print(f"Error details: {str(e)}")
        return documents

    def split_documents(self, documents, chunk_size=500, chunk_overlap=100):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        split_docs = text_splitter.split_documents(documents)
        print("Docs Split")
        return split_docs

    def create_embedding_list(self, split_docs):
        embed_list = [self.client.embeddings.create(input=[i.page_content], model="text-embedding-3-small").data[0].embedding for i in split_docs]
        return embed_list
    
    def create_metatdata_list(self, split_docs):
        meta_list = [{"video_id": i.metadata["source"], "text": i.page_content} for i in split_docs]
        return meta_list
    
    def create_ids(self, split_docs):
        ids = [str(i) for i in range(0, len(split_docs))]
        return ids
    
    def create_zip_file(self, split_docs):
        ids = self.create_ids(split_docs)
        embed_list = self.create_embedding_list(split_docs)
        meta_list = self.create_metatdata_list(split_docs)
        return zip(ids, embed_list, meta_list)
        

def langchain_embeddings(model_name = 'text-embedding-3-small'):
    embeddings = OpenAIEmbeddings(  
        model=model_name,  
        dimensions = 1536)  
    return embeddings

def langchain_vectorstore(index, embeddings):
    vectorstore = PineconeVectorStore(index, embeddings)  
    return vectorstore

def langchain_retriever(vectorstore):
    retriever = vectorstore.as_retriever()
    return retriever

def langchain_prompt():
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def langchain_model():
    # LLM
    model = ChatOpenAI(temperature=0.2, model = "gpt-3.5-turbo-0125")
    return model

def langchain_chain(retriever, prompt, model):
    chain = (
     RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser())
    return chain

def langchain_invoke(chain):
    query = str(input("What do you want to search on Youtube?"))
    try:
        result = chain.invoke(query)
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    helper = Helper(config_path)
    config = helper.load_config()
    pinecone_api = config["PINECONE_API_KEY"]
    openai_api = config["OPENAI_API_KEY"]
    #Create Pinecone Index
    pinecone = Pinecone(pinecone_api)
    index = pinecone.create_pinecone_index()
    #Prepare text for upsert
    prep_text_for_rag = PrepareTextForRag()
    blob_list = prep_text_for_rag.list_blobs_in_folder()
    blob_docs = prep_text_for_rag.load_blobs(blob_list)
    split_documents = prep_text_for_rag.split_documents(blob_docs)
    zip_file = prep_text_for_rag.create_zip_file(split_documents)
    #Upsert transcript
    pinecone.upsert_data(zip_file)

    #Langchain Chain
    embeddings = langchain_embeddings()
    vectorstore = langchain_vectorstore(index, embeddings)
    retriever = langchain_retriever(vectorstore)
    prompt = langchain_prompt()
    model = langchain_model()
    chain = langchain_chain(retriever, prompt, model)
    invoke = langchain_invoke(chain)

if __name__ == "__main__":
    main()