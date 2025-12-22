import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from langchain_huggingface import ChatHuggingFace
from langchain_pinecone import PineconeVectorStore

# Inizialization of the RAG components of the application
# (cached for efficiency)
@st.cache_resource
def setup_rag_components():
  # Retrieve tokens and API keys
  hf_token = os.environ.get("HF_TOKEN")
  pc_api_key = os.environ.get("PC_API_KEY")

  # If cannot retrieve credentials, stop the application
  if not hf_token or not pc_api_key:
    st.error("ERROR: Set HF_TOKEN and PC_API_KEY with your credentials.")
    st.stop()

  # Use a chat model from HuggingFace Inference Provider service
  llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=hf_token,
  )
  model = ChatHuggingFace(llm=llm)

  # Use an ebedding model from HuggingFace Inference Provider service
  embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=hf_token,
  )

  # Build a vector store using Pinecone
  vector_store = PineconeVectorStore(
    pinecone_api_key=os.environ.get("PC_API_KEY"),
    index_name='fast-api-learn-pages',
    embedding=embeddings
  )

  return model, vector_store

# Streamlit App UI
st.set_page_config(page_title="FastAPI RAG Bot", page_icon="🤖")
st.title("Fast API Learn Documentation RAG Application")
st.markdown("Ask me anything about how to get started working with FastAPI")

# Set-up the RAG components of the application
try:
  model, vector_store = setup_rag_components()
  st.success("✅ RAG components loaded successfully!")
except Exception as e:
  st.error(f"❌ Error initializing RAG components: {str(e)}")
  st.info("Make sure to set the correct credential as environment variables.")
  st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# Chat input
user_query = st.chat_input("Ask me something about FastAPI...")

# User sent a query
if user_query:
  # Add the new message to the chat history
  st.session_state.messages.append({"role": "user", "content": user_query})

  # Display user message
  with st.chat_message("user"):
    st.markdown(user_query)

  # Generate a response for the message
  with st.chat_message("assistant"):
    with st.spinner("Generating an answer..."):
      try:
        # RAG procedure
        ## Retrieve relevant documents
        relevant_docs = vector_store.similarity_search(user_query)

        ## Build context for the query
        context = "\n\n".join(
          f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
          for doc in relevant_docs
        )

        # Create the augmented prompt
        system_message = SystemMessage(content=(
          "You are a helpful assistant. Your task is to help software "
          "developers learn how to use FastAPI software. Include all the citations to "
          "sources inside your answers. To answer the queries use the following "
          "context:\n\n"
          f"{context}\n\n"
        ))

        user_message = HumanMessage(content=f"User query: {user_query}")

        # Get response from model
        messages = [system_message, user_message]
        response = model.invoke(messages)

        # Extract response content
        if hasattr(response, 'content'):
          response_text = response.content
        else:
          response_text = str(response)

        # Display response to the user
        st.markdown(response_text)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})

      except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Add a clear chat button in the sidebar
with st.sidebar:
  st.markdown("### Chat Controls")
  if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

  st.markdown("---")
  st.markdown("This Q&A bot will use the FastAPI documentation to answer your questions.")
  st.markdown("---")
  st.markdown("Built using LangChain, HuggingFace, Pinecone and Streamlit.")