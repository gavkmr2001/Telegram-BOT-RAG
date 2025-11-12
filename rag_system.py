import os
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
import fitz # PyMuPDF
import pickle
from functools import lru_cache

import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGSystem:
    def __init__(self):
        """Initializes the RAG system, loading the embedding model."""
        logging.info("Initializing RAG System...")
        # Load the sentence-transformer model directly
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.vector_store = None
        self.documents = []
        openai.api_key = config.OPENAI_API_KEY
        logging.info("RAG System Initialized.")

    def _load_documents(self):
        """Loads and chunks documents from the data directory."""
        logging.info(f"Loading documents from {config.DATA_PATH}...")
        all_texts = []
        doc_metadata = []

        for filename in os.listdir(config.DATA_PATH):
            if filename.endswith((".md", ".txt", ".pdf")):
                file_path = os.path.join(config.DATA_PATH, filename)
                
                # --- Simple Text Splitting Logic (Replaces LangChain's splitter) ---
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                # Split by paragraphs, then by sentences, then by words
                chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
                
                for i, chunk in enumerate(chunks):
                    all_texts.append(chunk)
                    doc_metadata.append({'source': filename, 'chunk_id': i})

        self.documents = [{"text": text, "metadata": meta} for text, meta in zip(all_texts, doc_metadata)]
        logging.info(f"Loaded and split documents into {len(self.documents)} chunks.")
        return all_texts

    def create_and_save_vector_store(self):
        """Creates embeddings and saves the FAISS index to disk."""
        texts = self._load_documents()
        if not texts:
            logging.error("No text chunks to process. Ingestion aborted.")
            return

        logging.info("Creating embeddings for all text chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        logging.info(f"Embeddings created with shape: {embeddings.shape}")

        # Create a FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype=np.float32))

        logging.info(f"Saving FAISS index to {config.VECTOR_STORE_PATH}...")
        faiss.write_index(index, config.VECTOR_STORE_PATH)
        
        # Save the document chunks themselves for later retrieval
        import pickle
        with open(config.VECTOR_STORE_PATH + ".pkl", "wb") as f:
            pickle.dump(self.documents, f)

        logging.info("Vector store created and saved successfully.")

    def load_vector_store(self):
        """Loads the FAISS index and documents from disk."""
        if not os.path.exists(config.VECTOR_STORE_PATH):
            logging.error(f"Vector store not found at {config.VECTOR_STORE_PATH}. Please run ingest.py first.")
            return False
            
        logging.info(f"Loading vector store from {config.VECTOR_STORE_PATH}...")
        self.vector_store = faiss.read_index(config.VECTOR_STORE_PATH)
        
        with open(config.VECTOR_STORE_PATH + ".pkl", "rb") as f:
            self.documents = pickle.load(f)
            
        logging.info(f"Vector store and {len(self.documents)} document chunks loaded successfully.")
        return True

    def _build_prompt(self, query, context_chunks):
        """Builds the prompt for the LLM with the retrieved context."""
        context = "\n\n---\n\n".join(context_chunks)
        
        prompt = f"""
You are a helpful assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer from the context provided, just say that you don't know. Do not try to make up an answer.

Context:
{context}

Question: {query}

Helpful Answer:"""
        return prompt

    def get_answer(self, query: str):
        """
        Retrieves an answer to a query using the RAG system.
        """
        if not self.vector_store:
            logging.error("Vector store is not loaded.")
            return "Error: The system is not ready. Please try again later.", []

        logging.info(f"Processing query: {query}")
        
        # 1. Embed the user's query
        query_embedding = self.embedding_model.encode([query])
        
        # 2. Perform similarity search in FAISS
        distances, indices = self.vector_store.search(np.array(query_embedding, dtype=np.float32), config.K_RETRIEVED_CHUNKS)
        
        # 3. Retrieve the actual text chunks and sources
        retrieved_chunks = [self.documents[i]['text'] for i in indices[0]]
        retrieved_sources = [self.documents[i]['metadata'] for i in indices[0]]
        
        # 4. Build the prompt
        prompt = self._build_prompt(query, retrieved_chunks)

        # 5. Call the OpenAI API for generation
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {e}")
            answer = "Sorry, I encountered an error while generating the answer."
        
        # Format sources for display
        sources = []
        unique_sources = {}
        for metadata in retrieved_sources:
            filename = metadata['source']
            if filename not in unique_sources:
                unique_sources[filename] = metadata['chunk_id']
                sources.append({"filename": filename, "snippet": self.documents[indices[0][0]]['text']})

        logging.info(f"Generated answer and found {len(sources)} unique source files.")
        return answer, sources