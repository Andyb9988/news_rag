import os
from logging import Logger
from newsapi import NewsApiClient
import json
from typing import List, Dict
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from utils.vector_database import PineconeHelper
from logging_utils.log_helper import get_logger

# logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger: Logger = get_logger(__name__)

news_api_key = os.getenv("NEWS_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api = os.getenv("PINECONE_API")

newsapi_client = NewsApiClient(api_key=news_api_key)
openai_client = OpenAI(api_key=openai_api_key)
pc_client = PineconeHelper(api_key=pinecone_api, index_name="all-news")


def get_content_dict(user_input: str, page: int = 1) -> List[Dict]:
    """
    Fetches articles from News API and returns a list of dictionaries containing content and metadata.
    """
    logger.info(f"Fetching articles for page {page}...")
    all_articles = newsapi_client.get_everything(
        q=f"{user_input}",
        sources="bbc-news,the-verge",
        domains="techcrunch.com, bbc.co.uk",
        from_param="2024-08-01",
        to="2024-08-02",
        language="en",
        sort_by="relevancy",
        page=page,
    )

    if all_articles["status"] == "ok":

        news_list = []
        for article in all_articles["articles"]:
            logger.info(
                f"Article: {article['title']} SOURCE: {article['source']['id']}, URL: {article['url']} now adding to a dictionary"
            )
            news_dict = {
                "content": article["content"],
                "metadata": {
                    "news_source": article["source"]["id"],
                    "url": article["url"],
                    "date_published": article["publishedAt"],
                    "title": article["title"],
                },
            }
            news_list.append(news_dict)
        return news_list
    else:
        raise ValueError(f"API request failed with status: {all_articles['status']}")


def process_and_chunk_articles(
    all_articles: List[Dict], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Dict]:
    """
    Processes and chunks articles using LangChain's RecursiveCharacterTextSplitter.
    """
    chunked_articles = []

    # Initialize the LangChain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    for article in all_articles:
        content = article["content"]
        metadata = article["metadata"]

        # Chunk the content using the LangChain text splitter
        chunks = text_splitter.split_text(content)
        logger.info(
            f"Chunking from {metadata['news_source']}, Title: {metadata['title']}"
        )

        for i, chunk in enumerate(chunks):
            chunked_article = {
                "id": str(uuid.uuid4()),
                "content": chunk,
                "metadata": {
                    "news_source": metadata["news_source"],
                    "url": metadata["url"],
                    "date_published": metadata["date_published"],
                    "title": metadata["title"],
                    "chunk_index": i,
                },
            }
            chunked_articles.append(chunked_article)

    return chunked_articles


def generate_embeddings(chunked_articles: List[Dict]) -> List[Dict]:
    """
    Generates embeddings for the content chunks using OpenAI's embedding model and creates a new dictionary.
    """
    embedded_articles = []

    for article in chunked_articles:
        content_chunk = article["content"]

        # Generate embedding for the content chunk
        logger.debug(
            f"Generating embedding for chunk with ID {article['id']} from title '{article['metadata']['title']}'"
        )
        response = (
            openai_client.embeddings.create(
                input=[content_chunk], model="text-embedding-3-small"
            )
            .data[0]
            .embedding
        )

        embedded_article = {
            "id": article["id"],
            "values": response,
            "metadata": {"content": content_chunk, **article["metadata"]},
        }

        embedded_articles.append(embedded_article)

    return embedded_articles


def main():
    page = 1
    all_chunked_articles = []
    max_pages = 5
    logger.info(f"Starting the process. Fetching up to {max_pages} pages of articles.")

    pc_client.pinecone_index()

    while page <= max_pages:
        try:
            news_list = get_content_dict(page=page)
            if not news_list:
                logger.info(f"No articles found on page {page}. Ending the process.")
                break

            chunked_articles = process_and_chunk_articles(news_list)
            all_chunked_articles.extend(chunked_articles)

            page += 1

        except Exception as e:
            logger.error(f"An error occurred on page {page}: {str(e)}")
            break

    if all_chunked_articles:

        logger.info(
            f"Generating embeddings for {len(all_chunked_articles)} chunks of articles."
        )
        embedded_articles = generate_embeddings(all_chunked_articles)

        logger.info("Upserting data to the Vector Database.")
        pc_client.upsert_data(embedded_articles)
    else:
        logger.warning(
            "No articles were processed. Skipping embedding and upserting steps."
        )

    logger.info("Process completed.")


if __name__ == "__main__":
    main()
