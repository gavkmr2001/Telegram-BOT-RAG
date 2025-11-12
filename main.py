import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

import config
from rag_system import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the RAG system
rag_system = RAGSystem()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /start command."""
    await update.message.reply_text(
        "Hello! I am your friendly RAG bot.\n"
        "Use /ask <your question> to ask me anything about our company policies.\n"
        "Use /help to see all commands."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /help command."""
    await update.message.reply_text(
        "Here are the available commands:\n"
        "/start - Welcome message\n"
        "/help - Show this help message\n"
        "/ask <your question> - Ask a question about our documents."
    )

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /ask command."""
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Please provide a question after the /ask command.")
        return

    await update.message.reply_text("Thinking...")
    
    answer, sources = rag_system.get_answer(query)

    # Format the response with sources
    response_message = f"**Answer:**\n{answer}\n\n"
    
    if sources:
        response_message += "**Sources:**\n"
        for i, source in enumerate(sources, 1):
            response_message += f"{i}. **{source['filename']}**\n"
            # Show a small snippet from the source
            snippet = source['snippet'].strip().replace('\n', ' ')
            response_message += f"   *...{snippet[:100]}...*\n"
            
    await update.message.reply_text(response_message, parse_mode='Markdown')


def main():
    """Main function to run the bot."""
    if not config.TELEGRAM_TOKEN:
        logging.error("TELEGRAM_BOT_TOKEN not found in .env file. Exiting.")
        return

    # Load the vector store on startup
    if not rag_system.load_vector_store():
        return # Stop if vector store loading fails

    app = ApplicationBuilder().token(config.TELEGRAM_TOKEN).build()

    # Add command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ask", ask))

    logging.info("Bot is starting up and polling for messages...")
    app.run_polling()

if __name__ == '__main__':
    main()