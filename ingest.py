from rag_system import RAGSystem

if __name__ == "__main__":
    print("Starting the ingestion process to build the vector store...")
    rag = RAGSystem()
    rag.create_and_save_vector_store()
    print("Ingestion complete. The vector store has been created and saved.")