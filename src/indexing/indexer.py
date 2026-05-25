import chromadb
from langchain_chroma import Chroma
from src.embeddings import create_embeddings
from src.indexing.chunker import load_law_documents, split_documents
from src import config
from src.timer import timer


@timer("build_index")
def build_index(force_recreate: bool = False):
    chroma_client = chromadb.HttpClient(
        host=config.CHROMA_HOST,
        port=config.CHROMA_PORT,
    )

    embeddings = create_embeddings()

    if force_recreate:
        try:
            chroma_client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
    )

    if force_recreate or vectorstore._collection.count() == 0:
        print("Loading and chunking documents...")
        docs = load_law_documents()
        chunks = split_documents(docs)
        print(f"Adding {len(chunks)} chunks to Chroma...")
        vectorstore.add_documents(chunks)
        print("Indexing complete.")
    else:
        print(f"Collection already has {vectorstore._collection.count()} documents.")

    return vectorstore


def get_vectorstore() -> Chroma:
    chroma_client = chromadb.HttpClient(
        host=config.CHROMA_HOST,
        port=config.CHROMA_PORT,
    )
    embeddings = create_embeddings()
    return Chroma(
        client=chroma_client,
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
    )
