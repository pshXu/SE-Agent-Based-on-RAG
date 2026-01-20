# src/rag/ingestion_llama.py
import os
# Fix for Hugging Face tokenizers deadlock warning/error in forked processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import logging
from typing import List, Dict, Any
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document as LlamaDocument,
)
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from pdf2image import convert_from_path, pdfinfo_from_path
import pytesseract
import fitz  # PyMuPDF

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import BGE_MODEL_NAME, DATA_RAW_PATH, COLLECTION_NAME

# Setup Logging
logging.basicConfig(level=logging.INFO)

class DualLanguagePDFReader(BaseReader):
    """
    Intelligent PDF Reader: 
    1. Tries fast text extraction first (PyMuPDF).
    2. Falls back to Dual-Lang OCR only if the PDF is a scanned image.
    """
    def load_data(self, file_path: str, extra_info: Dict[str, Any] = None) -> List[LlamaDocument]:
        file_name = os.path.basename(file_path)
        print(f"\nAnalyzing: {file_name}")
        
        try:
            # --- Phase 1: Try Fast Extraction ---
            doc = fitz.open(file_path)
            total_pages = doc.page_count
            extracted_pages = []
            
            is_scanned = True
            
            # Check the first few pages to determine if it's a scanned PDF
            check_pages = min(3, total_pages)
            total_text_len = 0
            for i in range(check_pages):
                page_text = doc[i].get_text().strip()
                total_text_len += len(page_text)
            
            # Heuristic: if average text per page > 50 chars, assume it's NOT a scan
            if total_text_len / check_pages > 50:
                print(f"  - Detected as native PDF. Extracting text directly...")
                is_scanned = False
                for i in range(total_pages):
                    page_text = doc[i].get_text()
                    metadata = extra_info.copy() if extra_info else {}
                    metadata.update({"page_label": str(i+1), "file_name": file_name, "mode": "extracted"})
                    extracted_pages.append(LlamaDocument(text=page_text, metadata=metadata))
                doc.close()
                return extracted_pages
            
            doc.close()
            
            # --- Phase 2: Fallback to OCR ---
            print(f"  - Detected as scanned PDF. Starting OCR (this will take time)...")
            ocr_documents = []
            
            for i in range(1, total_pages + 1):
                try:
                    # Convert single page
                    images = convert_from_path(file_path, first_page=i, last_page=i, dpi=150)
                    if not images: continue
                        
                    page_img = images[0]
                    # OCR
                    page_text = pytesseract.image_to_string(page_img, lang='chi_sim+eng')
                    
                    metadata = extra_info.copy() if extra_info else {}
                    metadata.update({"page_label": str(i), "file_name": file_name, "mode": "ocr"})
                    
                    ocr_documents.append(LlamaDocument(text=page_text, metadata=metadata))
                    
                    del images
                    del page_img
                    
                    if i % 10 == 0:
                        print(f"    - OCR processed page {i}/{total_pages}")
                        
                except Exception as page_e:
                    logging.warning(f"Error OCR-ing page {i}: {page_e}")
                    continue

            return ocr_documents
            
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            return []

def get_node_parser():
    return SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

def main():
    print("--- Starting Intelligent LlamaIndex Ingestion ---")
    
    # 1. Setup Embedding Model
    embed_model = HuggingFaceEmbedding(
        model_name=BGE_MODEL_NAME,
        embed_batch_size=16
    )
    Settings.embed_model = embed_model
    Settings.llm = None 
    Settings.node_parser = get_node_parser()

    # 2. Setup Vector Store
    persist_dir = os.path.join(project_root, "data", "llama_vector_store")
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Initialize Index
    index = None
    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        print("Loaded existing index from vector store.")
    except Exception:
        print("No existing index found. Creating new empty index.")
        index = VectorStoreIndex.from_documents([], storage_context=storage_context)

    # 4. Stream Process Files
    reader = DualLanguagePDFReader()
    
    file_count = 0
    for root, dirs, files in os.walk(DATA_RAW_PATH):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                
                try:
                    file_docs = reader.load_data(file_path)
                    if not file_docs: continue
                        
                    print(f"  - Inserting {len(file_docs)} pages into index...")
                    nodes = Settings.node_parser.get_nodes_from_documents(file_docs)
                    index.insert_nodes(nodes)
                    
                    # Persist metadata
                    index.storage_context.persist(persist_dir=persist_dir)
                    
                    file_count += 1
                    print(f"  - Completed: {file}")
                    
                except Exception as e:
                    logging.error(f"Failed to process {file}: {e}")
                    continue

    print(f"\n--- Ingestion Complete. Processed {file_count} files. ---")

if __name__ == "__main__":
    main()