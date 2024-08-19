import streamlit as st
from utils.vector_database import PineconeHelper
from utils.data_processor import process_and_chunk_articles, return_news_docs
from utils.news_api import GetNews
from config.config import PipelineConfiguration, get_pipeline_config
from logging import Logger
from logging_utils.log_helper import get_logger

logger: Logger = get_logger(__name__)
APP_CONFIG: PipelineConfiguration = get_pipeline_config()
index_name = APP_CONFIG.pc_index_name_news

news_client = GetNews()
pc_client = PineconeHelper(index_name=index_name)
pc_client.pinecone_index()

# Streamlit app UI
def main():
    page_config = {
        "page_title": "Search BBC News",
        "layout": "centered",
    }
    st.set_page_config(**page_config)
    st.markdown(
        "<h1 style='text-align: center;'> Ask The News Anything </h1>",
        unsafe_allow_html=True,
    )

    # Text input for search query
    input_query = st.text_input(
        "Enter a topic to search on BBC News and Tech Crunch e.g. 'Artificial Intelligence'"
    )

    # Button to trigger the process
    if st.button("Fetch and Process Articles"):
        with st.spinner("Fetching articles..."):
            try:
                # Fetch articles
                articles = []
                page = 1
                articles_page = news_client.get_content(input_query, page=page)
                articles.extend(articles_page)
                st.success(f"Fetched {len(articles)} articles.")

                # Process and chunk articles
                logger.info("Getting news documents")
                news = return_news_docs(articles_page)
                chunked_articles = process_and_chunk_articles(news)

                st.success(f"Chunked into {len(chunked_articles)} pieces.")

                # Upsert data into Pinecone
                pc_client.langchain_upload_documents_to_vdb(
                    chunked_articles, namespace=input_query
                )
                st.success("Data upserted into Pinecone successfully.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
