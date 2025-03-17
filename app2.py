import uuid
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from operator import itemgetter
import streamlit as st
import tempfile
from dotenv import load_dotenv
import os
import chromadb
from pathlib import Path
load_dotenv()
import pandas as pd

# openai_api_key = os.environ['OPENAI_API_KEY']

st.set_page_config(page_title="File QA Chatbot", page_icon="🤖")
st.title("Welcome to File QA Refrigeration and Cryogenics Chatbot 🤖")

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
  # Read documents
  docs = []
  temp_dir = tempfile.TemporaryDirectory()
  for file in uploaded_files:
    temp_filepath = os.path.join(temp_dir.name, file.name)
    with open(temp_filepath, "wb") as f:
      f.write(file.getvalue())
    loader = PyMuPDFLoader(temp_filepath)
    docs.extend(loader.load())

  # Split into documents chunks
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                                 chunk_overlap=200)
  doc_chunks = text_splitter.split_documents(docs)

  # Create document embeddings and store in Vector DB
  TMP_DIR = Path("./data").resolve().parent.joinpath("data", "tmp")
  LOCAL_VECTOR_STORE_DIR = Path("./data").resolve().parent.joinpath("data", "vector_stores")
  embeddings_model = OpenAIEmbeddings(openai_api_key = os.environ['OPENAI_API_KEY'])
  persist_directory = (LOCAL_VECTOR_STORE_DIR.as_posix() + "/" + "OpenAI_Embeddings")
  vectordb = Chroma.from_documents(doc_chunks, embeddings_model, persist_directory=persist_directory)

  # Define retriever object
  retriever = vectordb.as_retriever()
  return retriever

# Manages live updates to a Streamlit app's display by appending new text tokens
# to an existing text stream and rendering the updated text in Markdown
class StreamHandler(BaseCallbackHandler):
  def __init__(self, container, initial_text=""):
    self.container = container
    self.text = initial_text

  def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.text += token
    self.container.markdown(self.text)

# Creates UI element to accept PDF uploads
uploaded_files = st.sidebar.file_uploader(
  label="Upload PDF files", type=["pdf"],
  accept_multiple_files=True
)
if not uploaded_files:
  st.info("Please upload PDF documents to continue.")
  st.stop()

# Create retriever object based on uploaded PDFs
retriever = configure_retriever(uploaded_files)

# Load a connection to ChatGPT LLM
chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1,
                     streaming=True, openai_api_key = os.environ['OPENAI_API_KEY'],
                     max_tokens=1500)

# Create a prompt template for QA RAG System
qa_template = """
              Use only the following pieces of context to answer the question at the end.
              If you don't know the answer, just say that you don't know,
              don't try to make up an answer. Keep the answer as concise as possible.

              {context}

              Question: {question}
              """
qa_prompt = ChatPromptTemplate.from_template(qa_template)

# This function formats retrieved documents before sending to LLM
def format_docs(docs):
  return "\n\n".join([d.page_content for d in docs])

# Create a QA RAG System Chain
qa_rag_chain = (
  {
    "context": itemgetter("question") # based on the user question get context docs
      |
    retriever
      |
    format_docs,
    "question": itemgetter("question") # user question
  }
    |
  qa_prompt # prompt with above user question and context
    |
  chatgpt # above prompt is sent to the LLM for response
)

# Generate a session ID for each user
if "session_id" not in st.session_state:
  st.session_state["session_id"] = str(uuid.uuid4())

session_id = st.session_state["session_id"]

# Store conversation history in Streamlit session state with session ID
streamlit_msg_history = StreamlitChatMessageHistory(key=f"langchain_messages_{session_id}")

# Shows the first message when app starts
if len(streamlit_msg_history.messages) == 0:
  streamlit_msg_history.add_ai_message("Please ask your question?")

# Render current messages from StreamlitChatMessageHistory
for msg in streamlit_msg_history.messages:
  st.chat_message(msg.type).write(msg.content)

class PostMessageHandler(BaseCallbackHandler):
  def __init__(self, msg: st.write):
    BaseCallbackHandler.__init__(self)
    self.msg = msg
    self.sources = []

  def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
    source_ids = []
    for d in documents: # retrieved documents from retriever based on user query
      metadata = {
        "source": d.metadata["source"],
        "page": d.metadata["page"],
        "content": d.page_content[:200]
      }
      idx = (metadata["source"], metadata["page"])
      if idx not in source_ids: # store unique source documents
        source_ids.append(idx)
        self.sources.append(metadata)

  def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
    if len(self.sources):
      st.markdown("__Sources:__ "+"\n")
      st.dataframe(data=pd.DataFrame(self.sources[:3]),
                    width=1000)
      for source in self.sources[:3]:
        st.markdown(f"[{source['source']} - Page {source['page']}]({source['source']})") # Top 3 sources


# If user inputs a new prompt, display it and show the response
if user_prompt := st.chat_input():
  st.chat_message("human").write(user_prompt)
  streamlit_msg_history.add_user_message( user_prompt)  # Add user prompt to message history
  # This is where response from the LLM is shown
  with st.chat_message("ai"):
    # Initializing an empty data stream
    stream_handler = StreamHandler(st.empty())
    # UI element to write RAG sources after LLM response
    sources_container = st.write("")
    pm_handler = PostMessageHandler(sources_container)
    config = {"callbacks": [stream_handler, pm_handler]}
    # Get LLM response
    response = qa_rag_chain.invoke({"question": user_prompt}, config=config)
    streamlit_msg_history.add_ai_message( response.content)  # Add LLM response to message history

def generate_synthetic_qa(doc_chunks, num_pairs=5):
    """Generate synthetic QA pairs using ChatGPT from given document chunks."""
    qa_generator_prompt = ChatPromptTemplate.from_template("""
        Generate {num_pairs} question-answer pairs based on the provided document content.
        Keep the questions concise and relevant.

        Document:
        {document}

        Output format:
        Q1: <question>
        A1: <answer>
        Q2: <question>
        A2: <answer>
        ...
    """)

    qa_generator_chain = qa_generator_prompt | chatgpt | StrOutputParser()

    # Use the first document chunk as plain text
    sample_document = doc_chunks[0] if doc_chunks else "No document found."

    return qa_generator_chain.invoke({"document": sample_document, "num_pairs": num_pairs})


def evaluate_chatbot_accuracy():
    """Evaluates chatbot accuracy using synthetic QA pairs."""
    global retriever

    # Get document chunks from retriever
    doc_chunks = retriever.vectorstore.get()["documents"]

    if not doc_chunks or len(doc_chunks) == 0:
        st.error("No documents found in retriever.")
        return

    # Generate synthetic QA pairs
    synthetic_qa_pairs = generate_synthetic_qa(doc_chunks)

    st.write("### Synthetic QA Pairs Generated:")
    st.write(synthetic_qa_pairs)

# Streamlit UI button to trigger evaluation
if st.button("Evaluate Chatbot Accuracy"):
    evaluate_chatbot_accuracy()
