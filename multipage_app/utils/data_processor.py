import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import NewsURLLoader
from typing import List, Tuple, Dict, Union
from langchain_core.documents import Document
from logging import Logger
from logging_utils.log_helper import get_logger

logger: Logger = get_logger(__name__)


def return_news_docs(news_list: List[str]) -> List[Document]:
    logger.info(f"The news list recieved: {news_list}")
    loader = NewsURLLoader(urls=news_list)
    documents = loader.load()
    # create a list of page content and dictionary of metadata
    doc_list = []
    for doc in documents:
        if doc.metadata.get("publish_date"):
            doc.metadata["PublishedAt"] = doc.metadata["publish_date"].strftime(
                "%d-%m-%Y"
            )
            del doc.metadata["publish_date"]
        else:
            doc.metadata.pop("publish_date", None)
        del doc.metadata["language"]
        doc_list.append(doc)

    return doc_list


def process_and_chunk_articles(
    all_articles: List[Document], chunk_size: int = 1000, chunk_overlap: int = 250
) -> List[Document]:
    """
    Processes and chunks articles using LangChain's RecursiveCharacterTextSplitter.
    """
    chunked_articles = []

    # Initialize the LangChain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    for document in all_articles:
        content = document.page_content
        metadata = document.metadata

        # Chunk the content using the LangChain text splitter
        chunks = text_splitter.split_text(content)

        for i, chunk in enumerate(chunks):
            chunked_metadata = metadata.copy()  # Create a copy of the original metadata
            chunked_metadata.update(
                {
                    "chunk_index": i,
                    "title": metadata.get("title"),
                }
            )

            new_doc = Document(page_content=chunk, metadata=chunked_metadata)
            chunked_articles.append(new_doc)
    return chunked_articles
