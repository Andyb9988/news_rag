from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import NewsURLLoader
from langchain_core.documents import Document
from logging import Logger
from logging_utils.log_helper import get_logger

logger: Logger = get_logger(__name__)


def return_news_docs(news_list: List[str]) -> List[Document]:
    logger.info(f"Received news list with {len(news_list)} URLs.")

    if not news_list:
        logger.warning("No URLs provided in the news list.")
    try:
        loader = NewsURLLoader(urls=news_list)
        documents = loader.load()
        # create a list of page content and dictionary of metadata
        doc_list = []
        for doc in documents:
            # Process metadata
            publish_date = doc.metadata.get("publish_date")
            if publish_date:
                doc.metadata["PublishedAt"] = publish_date.strftime("%d-%m-%Y")
                del doc.metadata["publish_date"]
            else:
                doc.metadata.pop("publish_date", None)

            # Remove unnecessary metadata
            doc.metadata.pop("language", None)
            doc_list.append(doc)
        logger.info(f"Successfully processed {len(doc_list)} documents.")

        return doc_list

    except Exception as e:
        logger.exception(f"An error occurred while loading documents: {e}")
        raise RuntimeError("Failed to load news documents.") from e


def process_and_chunk_articles(
    all_articles: List[Document], chunk_size: int = 1000, chunk_overlap: int = 250
) -> List[Document]:
    """
    Processes and chunks articles using LangChain's RecursiveCharacterTextSplitter.
    """
    if not all_articles:
        logger.warning("No articles provided for processing and chunking.")
        return []

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
            chunked_metadata = metadata.copy()
            chunked_metadata.update(
                {
                    "chunk_index": i,
                    "title": metadata.get("title"),
                }
            )

            new_doc = Document(page_content=chunk, metadata=chunked_metadata)
            chunked_articles.append(new_doc)
        logger.info(
            f"Chunked {len(all_articles)} articles into {len(chunked_articles)} chunks."
        )
        return chunked_articles
