from logging import Logger
from typing import List, Dict
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logging_utils.log_helper import get_logger

logger: Logger = get_logger(__name__)


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
