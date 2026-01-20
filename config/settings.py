# config/settings.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Project Paths ---
# Get the root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the raw data directory
DATA_RAW_PATH = os.path.join(ROOT_DIR, "data", "raw", "books")

# Path to the vector store directory
VECTOR_STORE_PATH = os.path.join(ROOT_DIR, "data", "vector_store")

# --- RAG Model Configuration ---
# Name of the embedding model to use
# As per need.md, we use BAAI/bge-m3
BGE_MODEL_NAME = "BAAI/bge-m3"

# Name of the Cross-Encoder model for re-ranking
CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-base"

# --- Vector Store Configuration ---
# Name of the collection within ChromaDB
COLLECTION_NAME = "se_knowledge_base"

# --- LLM Configuration ---
# Example for OpenAI API Key, load from environment
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("--- Configuration Loaded ---")
print(f"Project Root: {ROOT_DIR}")
print(f"Raw Data Path: {DATA_RAW_PATH}")
print(f"Vector Store Path: {VECTOR_STORE_PATH}")
print(f"Embedding Model: {BGE_MODEL_NAME}")
print(f"Vector Collection Name: {COLLECTION_NAME}")
print("--------------------------")