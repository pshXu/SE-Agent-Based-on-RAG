# src/rag/vector_db.py
import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from config.settings import VECTOR_STORE_PATH, COLLECTION_NAME

def get_vector_store(embedding_function: Embeddings) -> Chroma:
    """
    Initializes and returns a Chroma vector store.

    This function sets up a persistent ChromaDB client using the specified
    path from settings. It creates or loads a collection with the specified
    name and returns a LangChain Chroma vector store instance ready for use.

    Args:
        embedding_function: The embedding function to use for the vector store.

    Returns:
        A LangChain Chroma vector store instance.
    """
    # Initialize a persistent ChromaDB client
    persistent_client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

    # Create or get the collection
    collection = persistent_client.get_or_create_collection(COLLECTION_NAME)

    # Initialize the LangChain Chroma vector store
    vector_store = Chroma(
        client=persistent_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    return vector_store

def get_retriever(embedding_function: Embeddings, search_kwargs={"k": 5}):
    """
    Initializes and returns a retriever from the Chroma vector store.

    Args:
        embedding_function: The embedding function to use for the vector store.
        search_kwargs: A dictionary of arguments to pass to the retriever's
                       search function (e.g., `{"k": 5}` to retrieve top 5).

    Returns:
        A retriever instance.
    """
    vector_store = get_vector_store(embedding_function=embedding_function)
    return vector_store.as_retriever(search_kwargs=search_kwargs)

if __name__ == '__main__':
    # This is an example of how to use this module
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    from config.settings import BGE_MODEL_NAME

    print("Initializing embedding function...")
    # Initialize the embedding function using the model name from settings
    embed_func = HuggingFaceEmbeddings(
        model_name=BGE_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Use CPU for embedding
        encode_kwargs={'normalize_embeddings': True}
    )

    print("Getting vector store...")
    # Get the vector store instance
    db = get_vector_store(embedding_function=embed_func)
    
    # Example: Add a document to the vector store
    # print("Adding a test document...")
    # db.add_texts(
    #     texts=["This is a test document about software engineering."],
    #     metadatas=[{"source": "test_document"}],
    #     ids=["test_id_1"]
    # )
    # print("Test document added.")

    # Verify the number of items in the collection
    collection_count = db._collection.count()
    print(f"The collection '{COLLECTION_NAME}' now contains {collection_count} documents.")

    # Example: Perform a similarity search
    if collection_count > 0:
        query = "What is software engineering?"
        print(f"\nPerforming similarity search for: '{query}'")
        search_results = db.similarity_search(query, k=1)
        
        if search_results:
            print("Found results:")
            for doc in search_results:
                print(f"  - Source: {doc.metadata.get('source', 'N/A')}")
                print(f"    Content: {doc.page_content[:150]}...")
        else:
            print("No results found.")
    else:
        print("\nSkipping similarity search because the collection is empty.")