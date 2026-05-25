import time
import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(
    host="localhost",
    port=8100,
    settings=Settings(chroma_client_auth_credentials="", anonymized_telemetry=False)
)

while True:
    time_1 = time.time()
    collections = client.list_collections()
    time_2 = time.time()
    print(f"Time to list collections: {time_2 - time_1} seconds")

    for c in collections:
        count = c.count()
        print(f"{c.name}: {count} chunks")
    
    time_3 = time.time()
    print(f"Time to count chunks: {time_3 - time_2} seconds")
    print("-----------------")
    user_input = input("Press Enter to continue...")
    print("-----------------")
    if user_input.lower() == "exit":
        break
