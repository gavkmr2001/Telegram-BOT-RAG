import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Bot and API Keys ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- RAG System Configuration ---
# The local model to use for creating embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Vector Store Configuration ---
# Directory to save the FAISS index
VECTOR_STORE_PATH = "vector_store/faiss_index.bin"
# Directory containing the knowledge base documents
DATA_PATH = "data/"

# --- Text Processing Configuration ---
# Parameters for splitting documents into chunks
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Retrieval Configuration ---
# Number of relevant chunks to retrieve to answer a question
K_RETRIEVED_CHUNKS = 4

# Number of past interactions to keep for conversational context
MESSAGE_HISTORY_LENGTH = 3
