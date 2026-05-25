from dotenv import load_dotenv
import os

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")

LLM_THINKING_ENABLED = os.getenv("LLM_THINKING_ENABLED", "true").lower() == "true"
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "high")

CHROMA_HOST = "localhost"
CHROMA_PORT = 8100
COLLECTION_NAME = "laws_rag"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
RETRIEVAL_K = 5
LAWS_DIR = "laws"

REFERENCE_PREVIEW_CHARS = 100
