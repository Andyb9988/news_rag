import streamlit as st
from utils.rag import LangchainAssistant
from utils.vector_database import PineconeHelper
from utils.get_embeddings import GenerateEmbeddings
from utils.streamlit_frontend import (
    format_docs,
    format_docs_metadata,
    clear_chat_history,
)
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config.config import PipelineConfiguration, get_pipeline_config
from logging_utils.log_helper import get_logger
from logging import Logger

APP_CONFIG: PipelineConfiguration = get_pipeline_config()
logger: Logger = get_logger(__name__)


index_name = APP_CONFIG.pc_index_name_news
if not index_name:
    logger.error(
        "Index name is not configured. Please set 'pc_index_name_news' in the configuration."
    )
    st.stop()

lc_client = LangchainAssistant(index_name=index_name)
pc_client = PineconeHelper(index_name=index_name)
embed_client = GenerateEmbeddings()

embedding = embed_client.langchain_embedding_model()
prompt = lc_client.summarise_prompt()
model = lc_client.langchain_model()


def chain():
    rag_chain = (
        RunnableParallel(
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        )
        | lc_client.summarise_prompt()
        | model
        | StrOutputParser()
    )
    return rag_chain


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = pc_client.langchain_pinecone_vectorstore(
        embeddings=embedding
    )
if "retriever" not in st.session_state:
    st.session_state.retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 5, "namespace": "AI"}
    )
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = chain()


def main():
    page_config = {
        "page_title": "Summarise The News",
        "layout": "centered",
        "initial_sidebar_state": "auto",
    }
    st.set_page_config(**page_config)
    st.markdown(
        "<h1 style='text-align: center;'> Summarise the News with Gaz </h1>",
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "system",
                "content": "Hello, I am Gaz, your helpful news summariser assistant. Please start by asking a question.😆",
            }
        ]

    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    response = ""

    if prompt := st.chat_input("Ask your question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)

        docs = st.session_state.retriever.invoke(prompt)
        logger.info(f"Retrieved docs: {docs}")
        formatted_docs = format_docs(docs)
        chain_input = {"context": formatted_docs, "question": prompt}
        response = st.session_state.rag_chain.invoke(chain_input)
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        st.sidebar.header("News References")
        metadata_formatted = format_docs_metadata(docs)
        for i, doc in enumerate(metadata_formatted):
            st.sidebar.markdown(f"***{i+1}***")
            st.sidebar.markdown(f"***Title:*** {doc['title']}")
            st.sidebar.markdown(f"***Link:*** [{doc['link']}]({doc['link']})")
            st.sidebar.divider()


if __name__ == "__main__":
    main()
