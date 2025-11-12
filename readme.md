# Mini-RAG Telegram Bot

This project is a lightweight but powerful Generative AI chatbot built for Telegram. It implements a Retrieval-Augmented Generation (RAG) system to answer user questions based on a private knowledge base of documents. The entire system is designed to run locally with minimal dependencies, showcasing a modular and efficient architecture.

The bot is capable of understanding user queries in natural language, retrieving relevant information from a set of Markdown documents, and generating accurate, context-aware answers. It also includes several enhancements like conversational memory, caching, and source citation.

## 🚀 Features

-   **Telegram Bot Interface**: Interact with the RAG system through a simple and familiar Telegram chat interface.
-   **Retrieval-Augmented Generation (RAG)**: Answers questions using a local knowledge base, ensuring responses are grounded in provided documents and not just the LLM's general knowledge.
-   **Local Embeddings**: Uses a lightweight `sentence-transformers` model to generate vector embeddings locally, ensuring data privacy and reducing API costs.
-   **Efficient Vector Search**: Employs `FAISS` (from Meta AI) for high-speed similarity search to find the most relevant document chunks.
-   **High-Quality Generation**: Leverages the OpenAI API (`gpt-3.5-turbo`) to synthesize final answers, demonstrating the ability to integrate with industry-standard models.
-   **Source Snippets**: Each answer is accompanied by citations from the source documents, allowing users to verify the information.
-   **Conversational Memory**: The bot maintains an awareness of the last 3 interactions per user, allowing it to understand and answer follow-up questions.
-   **Multi-Layer Caching**: Implements a dual-caching system for both embeddings and final answers to improve performance and reduce redundant computations for repeated queries.

## 🏛️ System Design & Architecture

The project is architected with a clear separation of concerns, dividing the process into two main phases: Ingestion and Inference.

### Phase 1: Ingestion (Offline Processing)

A one-time script (`ingest.py`) prepares the knowledge base.
1.  **Load Documents**: Loads all `.md`, `.txt`, and `.pdf` files from the `/data` directory.
2.  **Chunk Documents**: Splits the documents into smaller, manageable chunks for effective embedding.
3.  **Generate Embeddings**: Uses the local `all-MiniLM-L6-v2` model to convert each text chunk into a vector embedding.
4.  **Create & Save Vector Store**: Stores these embeddings in a `FAISS` index file (`vector_store/faiss_index.bin`) for fast retrieval. The raw text chunks are also saved in a corresponding `.pkl` file.

### Phase 2: Inference (Real-Time Bot Operation)

The main bot application (`main.py`) handles user interactions in real-time.

**Data Flow:**
[User on Telegram] --/ask--> [main.py: Telegram Bot]
|
v
[RAGSystem: get_answer(query, history)]
/
v v
[Vector Store (FAISS)] <--Similarity Search-- [Embedding Model (Cached)]
|
v
[Retrieved Context] --> [Prompt Template w/ History] --> [OpenAI LLM] --> [Final Answer]
|
v
[main.py: Sends Reply w/ Sources] --> [User on Telegram]
code
Code
## 🛠️ Tech Stack

-   **Bot Framework**: `python-telegram-bot`
-   **Embedding Model**: `sentence-transformers` (`all-MiniLM-L6-v2`)
-   **Vector Store**: `faiss-cpu` (from Meta AI)
-   **LLM (Generation)**: `OpenAI API` (`gpt-3.5-turbo`)
-   **Configuration**: `python-dotenv`
-   **Dependencies**: `NumPy`, `PyMuPDF`

## ⚙️ Setup and Installation

Follow these steps to run the bot locally on your machine.

### Prerequisites

-   Python 3.8+
-   A Telegram Bot Token
-   An OpenAI API Key

### 1. Clone the Repository & Set Up Environment

```bash
# Clone the repository (or just create the project folder)
git clone https://your-repo-url.com/rag-telegram-bot.git
cd rag-telegram-bot

# Create and activate a Python virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
2. Install Dependencies
Install all the required Python libraries using the requirements.txt file.
code
Bash
pip install -r requirements.txt
3. Configure API Keys
Create a file named .env in the root of the project directory and add your secret keys.
code
Code
TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN_HERE"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
4. Prepare the Knowledge Base
Place your knowledge base documents (e.g., .md, .txt, or .pdf files) inside the /data directory.
5. Run the Ingestion Process (One Time Only)
Run the ingestion script to process your documents and create the vector store. This must be done once before you can run the bot.
code
Bash
python ingest.py
This will create a faiss_index.bin file and a faiss_index.bin.pkl file inside the /vector_store directory.
6. Run the Bot
You are now ready to start the Telegram bot.
code
Bash
python main.py
The terminal will show a log message indicating that the bot is running and polling for messages.
🤖 How to Use the Bot
Open your Telegram app and interact with your bot using the following commands:
/start: Displays a welcome message.
/help: Shows a list of available commands.
/ask <your question>: Ask a question related to the documents in your knowledge base.
Example Interaction:
User: /ask how many days of sick leave do we get?
Bot (Thinking...):
Bot (Final Answer):
Answer:
We offer 10 days of paid sick leave per year. A doctor's note is required for absences of more than 3 consecutive days.
Sources:
policy_on_leave.md
...We offer 10 days of paid sick leave per year. A doctor's note is required for absences longer than 3 co...