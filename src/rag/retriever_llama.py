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

import numpy as np

class LlamaIndexRetriever:
    """
    A wrapper around LlamaIndex to make it compatible with our existing
    LangChain-based agent architecture. Supports Hybrid Search (RRF) and MMR Re-ranking.
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
            # A. Vector Retriever - Fetch more for better RRF fusion
            vector_retriever = index.as_retriever(similarity_top_k=50)
            
            # B. BM25 Retriever - Fetch more for better RRF fusion
            logging.info("Building BM25 index from docstore...")
            nodes = list(storage_context.docstore.docs.values())
            if not nodes:
                logging.warning("No nodes found in docstore. BM25 will be empty.")
                bm25_retriever = None
            else:
                bm25_retriever = BM25Retriever.from_defaults(
                    nodes=nodes,
                    similarity_top_k=50,
                    language="en" 
                )
            
            # 5. Create Fusion Retriever (RRF)
            # It will pick the best 30 from the 100 combined candidates
            retrievers = [vector_retriever]
            if bm25_retriever:
                retrievers.append(bm25_retriever)
                
            self._fusion_retriever = QueryFusionRetriever(
                retrievers=retrievers,
                similarity_top_k=30, # Final candidates for Cross-Encoder
                num_queries=1, 
                mode="reciprocal_rerank",
                use_async=False,
                verbose=True
            )
            
            logging.info("LlamaIndex Hybrid Retriever initialized.")
            
        except Exception as e:
            logging.error(f"Failed to initialize LlamaIndex: {e}")
            raise e

    def rerank_documents(self, query: str, documents: List[Document], k: int = 5, mmr_lambda: float = 0.5) -> List[Document]:
        """
        Global Rerank with MMR (Maximal Marginal Relevance).
        Combines Cross-Encoder precision with diversity filtering.
        """
        if not documents:
            return []
            
        # 1. Calculate Relevance Scores (Sim1) using Cross-Encoder
        pairs = [[query, doc.page_content] for doc in documents]
        raw_relevance_scores = self.cross_encoder.predict(pairs)
        
        # --- 核心优化：Sigmoid 归一化 ---
        # 将无界的 Logits 压缩到 [0, 1] 区间，与向量相似度对齐
        relevance_scores = 1 / (1 + np.exp(-raw_relevance_scores))
        
        # 2. Calculate Diversity (Sim2) using Embeddings
        # Get embeddings for all candidate documents
        doc_texts = [doc.page_content for doc in documents]
        doc_embeddings = Settings.embed_model.get_text_embedding_batch(doc_texts)
        doc_embeddings = np.array(doc_embeddings)
        
        # 3. MMR Iterative Selection
        selected_indices = []
        # 只保留相关性分数大于阈值 (如 0.01) 的候选者，防止选入完全无关的内容
        candidate_indices = [i for i, score in enumerate(relevance_scores) if score > 0.01]
        
        if not candidate_indices:
            return documents[:k] # 如果都太差，退化为原始排序的前 K 个
            
        # Always pick the most relevant one first from our candidates
        first_idx = candidate_indices[np.argmax(relevance_scores[candidate_indices])]
        selected_indices.append(first_idx)
        candidate_indices.remove(first_idx)
        
        while len(selected_indices) < k and candidate_indices:
            mmr_scores = []
            for char_idx in candidate_indices:
                # Part 1: Relevance Score (Now strictly 0-1)
                rel_score = relevance_scores[char_idx]
                
                # Part 2: Max Similarity with already selected documents
                target_emb = doc_embeddings[char_idx]
                selected_embs = doc_embeddings[selected_indices]
                
                # Cosine similarity = dot product of normalized embeddings
                similarities = np.dot(selected_embs, target_emb) / (
                    np.linalg.norm(selected_embs, axis=1) * np.linalg.norm(target_emb) + 1e-9
                )
                max_sim = np.max(similarities)
                
                # MMR Formula: lambda * relevance - (1 - lambda) * redundancy
                mmr_val = mmr_lambda * rel_score - (1 - mmr_lambda) * max_sim
                mmr_scores.append((mmr_val, char_idx))
            
            # Pick the best MMR score
            best_idx = max(mmr_scores, key=itemgetter(0))[1]
            selected_indices.append(best_idx)
            candidate_indices.remove(best_idx)
            
        return [documents[i] for i in selected_indices]

    def retrieve(self, query: str, k_final: int = 5) -> List[Document]:
        """
        Retrieves documents using Hybrid Search (RRF) + Re-ranking.
        Now optimized for Markdown structured chunks.
        """
        if not self._fusion_retriever:
            raise ValueError("Retriever not initialized.")
            
        # 1. Hybrid Retrieval (RRF)
        # Fetch high-quality candidates from both Vector and BM25 indices
        nodes = self._fusion_retriever.retrieve(query)
        
        # 2. Convert to LangChain Documents for Agent compatibility
        # We no longer need MetadataReplacement because Markdown chunks are 
        # already semantically complete.
        candidate_docs = []
        for node_with_score in nodes:
            node = node_with_score.node
            metadata = node.metadata.copy() if node.metadata else {}
            metadata["score"] = node_with_score.score
            
            doc = Document(
                page_content=node.get_content(), # Standard content for Markdown nodes
                metadata=metadata
            )
            candidate_docs.append(doc)
            
        # 3. Perform Cross-Encoder Re-ranking to pick the best Top-K
        return self.rerank_documents(query, candidate_docs, k=k_final)
