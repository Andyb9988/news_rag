import streamlit as st
from utils.rag import LangchainAssistant
from utils.vector_database import PineconeHelper
from utils.get_embeddings import GenerateEmbeddings
from logging_utils.log_helper import get_logger
from logging import Logger
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser

logger: Logger = get_logger(__name__)
index_name = "all-news"

lc_client = LangchainAssistant(index_name=index_name)
pc_client = PineconeHelper(index_name=index_name)
embed_client = GenerateEmbeddings()

embedding = embed_client.langchain_embedding_model()
vectorstore = pc_client.langchain_pinecone_vectorstore(embeddings=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10, "namespace": "AI"})
prompt = lc_client.summarise_prompt()
model = lc_client.langchain_model()


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def chain():
    rag_chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | lc_client.summarise_prompt()
        | model
        | StrOutputParser()
    )
    return rag_chain


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

    docs = retriever.invoke(prompt)
    formatted_docs = format_docs(docs)

    rag_chain = chain()
    response = rag_chain.invoke(prompt)
    # "context": formatted_docs, "question": prompt}
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
