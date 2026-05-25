from langchain_openai import OpenAIEmbeddings
from src import config


def create_embeddings():
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.EMBEDDING_BASE_URL,
        api_key=config.EMBEDDING_API_KEY,
        chunk_size=32,
        check_embedding_ctx_length=False,
    )
