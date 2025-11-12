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
        """Initializes the RAG system, loading the embedding model and setting up caches."""
        logging.info("Initializing RAG System...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.vector_store = None
        self.documents = []
        openai.api_key = config.OPENAI_API_KEY
        
        # --- NEW: Caching Mechanisms ---
        # Cache for query embeddings to avoid re-calculating
        self.embedding_cache = {}
        # Simple cache for final answers to avoid re-running the whole chain for the same query
        self.answer_cache = {}

        logging.info("RAG System Initialized.")

    # --- NEW: Caching for Embeddings ---
    def _get_query_embedding(self, query: str):
        """Gets the embedding for a query, using a cache to avoid re-computation."""
        if query in self.embedding_cache:
            logging.info(f"Embedding cache hit for query: '{query}'")
            return self.embedding_cache[query]
        
        logging.info(f"Embedding cache miss. Computing embedding for query: '{query}'")
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
If you don't know the answer from the context, just say that you don't know. Do not try to make up an answer.

{history_str}

Context from documents:
{context}

Question: {query}

Helpful Answer:"""
        return prompt

    def get_answer(self, query: str, history: list = None):
        """
        Retrieves an answer to a query using the RAG system, now with history and caching.
        """
        # --- NEW: Answer Caching ---
        if query in self.answer_cache:
            logging.info(f"Answer cache hit for query: '{query}'")
            return self.answer_cache[query]

        if not self.vector_store:
            logging.error("Vector store is not loaded.")
            return "Error: The system is not ready. Please try again later.", []

        logging.info(f"Processing query with history: {query}")
        
        # 1. Embed the user's query using the new cached method
        query_embedding = self._get_query_embedding(query)
        
        # 2. Perform similarity search
        distances, indices = self.vector_store.search(np.array(query_embedding, dtype=np.float32), config.K_RETRIEVED_CHUNKS)
        
        # 3. Retrieve chunks and sources
        retrieved_chunks = [self.documents[i]['text'] for i in indices[0]]
        retrieved_sources = [self.documents[i]['metadata'] for i in indices[0]]
        
        # 4. Build the prompt with history
        prompt = self._build_prompt(query, retrieved_chunks, history)

        # 5. Call OpenAI API
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
        
        # Format sources for display (Refined to be more robust)
        sources = []
        if retrieved_sources:
            # Use a set to only show unique source files
            unique_filenames = set()
            for i, metadata in enumerate(retrieved_sources):
                filename = metadata.get('source', 'Unknown Source')
                if filename not in unique_filenames:
                    sources.append({
                        "filename": filename,
                        "snippet": self.documents[indices[0][i]]['text']
                    })
                    unique_filenames.add(filename)

        logging.info(f"Generated answer and found {len(sources)} unique source files.")
        
        # --- NEW: Store result in answer cache ---
        result = (answer, sources)
        self.answer_cache[query] = result
        
        return result
