# src/rag/retriever_llama.py
import os
import sys
import logging
from typing import List
from operator import itemgetter
from langchain_core.documents import Document

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage
)
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import get_response_synthesizer
import chromadb
from sentence_transformers import CrossEncoder

from config.settings import BGE_MODEL_NAME, COLLECTION_NAME, CROSS_ENCODER_MODEL_NAME

# Setup Logging
logging.basicConfig(level=logging.INFO)

class LlamaIndexRetriever:
    """
    A wrapper around LlamaIndex to make it compatible with our existing
    LangChain-based agent architecture. Supports Hybrid Search (RRF) and Re-ranking.
    """
    
    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self._fusion_retriever = None
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        self._init_index()
        
    def _init_index(self):
        """Initializes the LlamaIndex retrievers (Vector + BM25) from disk."""
        logging.info("Initializing LlamaIndex Hybrid Retriever...")
        
        # 1. Setup Embedding
        embed_model = HuggingFaceEmbedding(model_name=BGE_MODEL_NAME)
        Settings.embed_model = embed_model
        Settings.llm = None 
        
        # 2. Setup Vector Store
        db = chromadb.PersistentClient(path=self.persist_dir)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # 3. Load Storage Context (contains DocStore for BM25)
        try:
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, 
                persist_dir=self.persist_dir
            )
            # Load Index
            index = load_index_from_storage(storage_context)
            
            # 4. Create Retrievers
            # A. Vector Retriever
            vector_retriever = index.as_retriever(similarity_top_k=10)
            
            # B. BM25 Retriever
            # We need to extract nodes from the docstore to build BM25
            # Note: This loads all nodes into memory. Fine for small/medium datasets.
            logging.info("Building BM25 index from docstore...")
            nodes = list(storage_context.docstore.docs.values())
            if not nodes:
                logging.warning("No nodes found in docstore. BM25 will be empty.")
                bm25_retriever = None
            else:
                bm25_retriever = BM25Retriever.from_defaults(
                    nodes=nodes,
                    similarity_top_k=10,
                    stemmer=None, # Use default or configure based on language
                    language="en" # Default, adjust if needed for Chinese specific stemmer
                )
            
            # 5. Create Fusion Retriever (RRF)
            retrievers = [vector_retriever]
            if bm25_retriever:
                retrievers.append(bm25_retriever)
                
            self._fusion_retriever = QueryFusionRetriever(
                retrievers=retrievers,
                similarity_top_k=20, # Fetch more for re-ranking
                num_queries=1, # We do decomposition outside, so just use the raw query here
                mode="reciprocal_rerank",
                use_async=False,
                verbose=True
            )
            
            logging.info("LlamaIndex Hybrid Retriever initialized.")
            
        except Exception as e:
            logging.error(f"Failed to initialize LlamaIndex: {e}")
            raise e

    def rerank_documents(self, query: str, documents: List[Document], k: int = 5) -> List[Document]:
        """
        Re-ranks an arbitrary list of documents against a query using Cross-Encoder.
        """
        if not documents:
            return []
            
        # Prepare pairs for Cross-Encoder
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        
        # Combine and sort
        scored_docs = sorted(zip(documents, scores), key=itemgetter(1), reverse=True)
        
        # Return top k
        return [doc for doc, score in scored_docs[:k]]

    def retrieve(self, query: str, k_final: int = 5) -> List[Document]:
        """
        Retrieves documents using Hybrid Search (RRF) + Sentence Window logic + Re-ranking.
        """
        if not self._fusion_retriever:
            raise ValueError("Retriever not initialized.")
            
        # 1. Hybrid Retrieval (RRF)
        # This returns NodeWithScore objects
        nodes = self._fusion_retriever.retrieve(query)
        
        # 2. Post-processing: Metadata Replacement (Sentence Window)
        # We need to manually apply this because QueryFusionRetriever might not chain postprocessors automatically
        postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")
        nodes = postprocessor.postprocess_nodes(nodes)
        
        # 3. Convert to LangChain Documents
        candidate_docs = []
        for node_with_score in nodes:
            node = node_with_score.node
            metadata = node.metadata.copy() if node.metadata else {}
            metadata["score"] = node_with_score.score
            
            doc = Document(
                page_content=node.get_text(), # This will now be the 'window' text
                metadata=metadata
            )
            candidate_docs.append(doc)
            
        # 4. Perform Cross-Encoder Re-ranking
        return self.rerank_documents(query, candidate_docs, k=k_final)
