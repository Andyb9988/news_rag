# import os
# from logging import Logger
# from newsapi import NewsApiClient
# import json
# from typing import List, Dict
# import uuid
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from multipage_app.utils.rag import LangchainAssistant
from multipage_app.utils.data_processor import return_news, process_and_chunk_articles
from multipage_app.utils.news_api import GetNews
from multipage_app.utils.vector_database import PineconeHelper
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

model_name = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=model_name, dimensions=1536)
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api = os.getenv("PINECONE_API")
news_api_key = os.getenv("NEWS_API_KEY")


def main():
    pc = PineconeHelper(index_name="all-news")
    get_news = GetNews()

    news_dict = get_news.get_content_dict("Artificial Intelligence")
    news = return_news(news_dict)
    chunked_news = process_and_chunk_articles(news)
    print([doc.metadata for doc in chunked_news])
    print([doc.page_content for doc in chunked_news])
    pc.langchain_upload_documents_to_vdb(chunked_news)


# pc = PineconeHelper(index_name='all-news')

# lc = LangchainAssistant(index_name="all-news")
# prompt = lc.summarise_prompt()
# model = ChatOpenAI(temperature=0.2, api_key = openai_api_key, model="gpt-4o-mini")
# vectorstore = PineconeVectorStore(index_name='all-news', embedding=embeddings, pinecone_api_key=pinecone_api)
# retriever = vectorstore.as_retriever()


# news = retriever.invoke("tell me the latest ai news?")
# print(news)
# retrieval_chain = (
#     {
#         "context":retriever,
#         "question": RunnablePassthrough(),
#     }
#     | prompt
#     | model
#     | StrOutputParser()
# )
# news = retrieval_chain.invoke("Tell me the latest AI News")
# for i, doc in enumerate(news):
#     print(f"Document {i+1}")
#    # print(f"Content: {doc.page_content}")
#     print(f"Metadata: {doc.metadata}")
#     print(doc)
# print(retrieval_chain.invoke("Tell me the latest AI News"))
# print(results)


if __name__ == "__main__":
    main()
