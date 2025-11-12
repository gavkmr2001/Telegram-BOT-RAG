import os
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
import pickle

import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGSystem:
    def __init__(self):
        """Initializes the RAG system, loading the embedding model and setting up caches."""
        logging.info("Initializing RAG System...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.vector_store = None
        self.documents = []
        openai.api_key = config.OPENAI_API_KEY
        
        # Caching mechanisms
        self.embedding_cache = {}
        self.answer_cache = {}
        logging.info("RAG System Initialized.")

    # --- INGESTION METHODS ---

    def _load_documents(self):
        """Loads and chunks documents from the data directory."""
        logging.info(f"Loading documents from {config.DATA_PATH}...")
        all_texts = []
        doc_metadata = []

        for filename in os.listdir(config.DATA_PATH):
            if filename.endswith((".md", ".txt")): # Simplified for .md and .txt for now
                file_path = os.path.join(config.DATA_PATH, filename)
                
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                # Simple chunking by paragraphs
                chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
                
                for i, chunk in enumerate(chunks):
                    all_texts.append(chunk)
                    doc_metadata.append({'source': filename, 'chunk_id': i})

        self.documents = [{"text": text, "metadata": meta} for text, meta in zip(all_texts, doc_metadata)]
        logging.info(f"Loaded and split documents into {len(self.documents)} chunks.")
        return all_texts

    def create_and_save_vector_store(self):
        """Creates embeddings and saves the FAISS index and documents to disk."""
        texts = self._load_documents()
        if not texts:
            logging.error("No text chunks to process. Ingestion aborted.")
            return

        logging.info("Creating embeddings for all text chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        logging.info(f"Embeddings created with shape: {embeddings.shape}")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype=np.float32))

        logging.info(f"Saving FAISS index to {config.VECTOR_STORE_PATH}...")
        faiss.write_index(index, config.VECTOR_STORE_PATH)
        
        with open(config.VECTOR_STORE_PATH + ".pkl", "wb") as f:
            pickle.dump(self.documents, f)

        logging.info("Vector store created and saved successfully.")

    def load_vector_store(self):
        """Loads the FAISS index and documents from disk."""
        index_path = config.VECTOR_STORE_PATH
        docs_path = config.VECTOR_STORE_PATH + ".pkl"

        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            logging.error(f"Vector store files not found. Please run ingest.py first.")
            return False
            
        logging.info(f"Loading vector store from {index_path}...")
        self.vector_store = faiss.read_index(index_path)
        
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)
            
        logging.info(f"Vector store and {len(self.documents)} document chunks loaded successfully.")
        return True

    # --- INFERENCE METHODS ---

    def _get_query_embedding(self, query: str):
        """Gets the embedding for a query, using a cache to avoid re-computation."""
        if query in self.embedding_cache:
            logging.info(f"Embedding cache hit for query: '{query}'")
            return self.embedding_cache[query]
        
        embedding = self.embedding_model.encode([query])
        self.embedding_cache[query] = embedding
        return embedding

    def _build_prompt(self, query, context_chunks, history):
        """Builds the prompt for the LLM with retrieved context and chat history."""
        context = "\n\n---\n\n".join(context_chunks)
        
        history_str = ""
        if history:
            history_str += "Here is the recent conversation history:\n"
            for user_msg, bot_msg in history:
                history_str += f"User: {user_msg}\nBot: {bot_msg}\n"
        
        prompt = f"""
You are a helpful assistant. Use the following pieces of context and the conversation history to answer the question at the end.
If you don't know the answer from the context provided, just say that you don't know. Do not try to make up an answer.

{history_str}

Context from documents:
{context}

Question: {query}

Helpful Answer:"""
        return prompt

    def get_answer(self, query: str, history: list = None):
        """
        Retrieves an answer to a query using the RAG system, with history and caching.
        """
        if query in self.answer_cache:
            logging.info(f"Answer cache hit for query: '{query}'")
            return self.answer_cache[query]

        if not self.vector_store:
            return "Error: The system is not ready. Please try again later.", []

        query_embedding = self._get_query_embedding(query)
        
        distances, indices = self.vector_store.search(np.array(query_embedding, dtype=np.float32), config.K_RETRIEVED_CHUNKS)
        
        retrieved_chunks = [self.documents[i]['text'] for i in indices[0]]
        retrieved_sources_metadata = [self.documents[i]['metadata'] for i in indices[0]]
        
        prompt = self._build_prompt(query, retrieved_chunks, history)

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
        
        sources = []
        unique_filenames = set()
        for i, metadata in enumerate(retrieved_sources_metadata):
            filename = metadata.get('source', 'Unknown Source')
            if filename not in unique_filenames:
                sources.append({
                    "filename": filename,
                    "snippet": self.documents[indices[0][i]]['text']
                })
                unique_filenames.add(filename)

        result = (answer, sources)
        self.answer_cache[query] = result
        
        return result