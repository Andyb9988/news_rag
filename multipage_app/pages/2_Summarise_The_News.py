import streamlit as st
from utils.rag import LangchainAssistant
from utils.vector_database import PineconeHelper
from logging_utils.log_helper import get_logger
from logging import Logger
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger: Logger = get_logger(__name__)
index_name = "all-news"

lc_assistant = LangchainAssistant(index_name=index_name)
pc_client = PineconeHelper(index_name=index_name)


def initialise_langchain():
    # Generate embeddings and setup vectorstore
    embeddings = lc_assistant.langchain_embeddings()
    vectorstore = lc_assistant.langchain_vectorstore(embeddings)

    # Model and retriever setup
    model = lc_assistant.langchain_model()
    retriever = lc_assistant.langchain_retriever(vectorstore)
    # Prompt template setup
    prompt = lc_assistant.summarise_prompt()

    logger.info(f"Model: {model}, Retriever: {retriever}, prompt: {prompt}")
    return model, retriever, prompt


model, retriever, prompt = initialise_langchain()


def rag_chain():
    # Assuming RunnablePassthrough, model, and StrOutputParser are callable and can be chained
    context_question = {
        "context": RunnablePassthrough(),
        "question": RunnablePassthrough(),
    }

    # Apply the prompt to the context_question processing
    processed_context_question = model(context_question | prompt)

    # Parse the output string
    parsed_output = StrOutputParser(processed_context_question)

    return parsed_output


page_config = {
    "page_title": "Summarise The News",
    "layout": "centered",
    "initial_sidebar_state": "auto",
}
st.set_page_config(**page_config)
st.markdown(
    "<h1 style='text-align: center;'> Ask Youtube Anything </h1>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <h2> Welcome to a simple RAG APP </h2>
        """,
        unsafe_allow_html=True,
    )


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
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

    news = retriever.invoke(prompt)
    for i, doc in enumerate(news):
        logger.info(f"Document {i+1}")
        logger.info(f"Content: {doc.page_content}")
        logger.info(f"Metadata: {doc.metadata}")

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    rag = rag_chain()
    response = rag.invoke({"context": news, "question": prompt})

    #  response = lc_assistant.langchain_streamlit_invoke(prompt=prompt, chain=chain)
    with st.chat_message("assistant"):
        response

    st.session_state.messages.append({"role": "assistant", "content": response})
