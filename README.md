# MenuData AI Chatbot

## Table of Contents
- [Overview](#overview)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Key Features](#key-features)
- [Data Ingestion & Indexing](#data-ingestion--indexing)
- [Query Processing Pipeline](#query-processing-pipeline)
- [LLM Integration & Prompt Engineering](#llm-integration--prompt-engineering)
- [Factual Consistency & References](#factual-consistency--references)
- [Scalability & Future Extensions](#scalability--future-extensions)
- [Sample Queries](#sample-queries)

## Overview

MenuData AI Chatbot is an intelligent assistant designed to provide information about restaurants, menus, food history, and more. It combines a powerful backend with a user-friendly frontend to deliver a seamless chatbot experience.

## Technical Architecture

The system consists of two main components:

1. **Backend (src/bot_backend.py)**: A FastAPI application that handles query processing, data retrieval, and LLM integration.
2. **Frontend (src/frontend.py)**: A Streamlit-based user interface for interacting with the chatbot.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/menudata-ai-chatbot.git
   cd menudata-ai-chatbot
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   OPENAI_API_KEY=your_openai_api_key
   DATABASE_URL=your_postgres_database_url
   SERPAPI_KEY=your_serpapi_key
   ```

5. Run the backend:
   ```
   python bot_backend.py
   ```

6. In a new terminal, run the frontend:
   ```
   streamlit run frontend.py
   ```

## Key Features

### Query Type Classification

The system uses an LLM to classify incoming queries into multiple categories:
- Greeting
- SQL (database queries)
- Wikipedia (general knowledge)
- GoogleSearch (recent or specific information)
- VectorDBsearch (food history from PDF)
- Irrelevant

This classification allows the system to determine which data sources to query and how to process the information.

### One-Shot Prompt Engineering

The system employs one-shot prompt engineering techniques to guide the LLM in various tasks:

1. **SQL Query Generation**: A structured prompt is used to convert natural language queries into SQL.
2. **Query Classification**: A prompt with examples helps the LLM categorize queries accurately.
3. **Follow-up Detection**: A specialized prompt determines if a query is a follow-up to previous conversation.

### Response Merging

The system combines information from multiple sources to generate comprehensive responses:

1. Retrieves relevant data from each classified source (SQL, Wikipedia, Google, Vector DB).
2. Concatenates the retrieved information into a single prompt.
3. Uses the LLM to generate a coherent response based on the combined data.

This approach ensures that the final response is informative, context-aware, and draws from diverse sources.

### Data Ingestion & Indexing

The system ingests data from multiple sources:

1. **Restaurant Database**: A PostgreSQL database containing restaurant information, menu items, and ingredients[1].

2. **PDF Document**: The system loads a PDF file ("evolution_of_american_food.pdf") and indexes its content using vector embeddings[1].

3. **External Sources**: Wikipedia and Google Search results are fetched dynamically for relevant queries[1].

Data ingestion and indexing process:

1. The PDF document is loaded using `PyPDFLoader` and split into chunks using `RecursiveCharacterTextSplitter`[1].
2. Text chunks are embedded using OpenAI's `text-embedding-ada-002` model[1].
3. Embeddings are stored in an in-memory vector database (`DocArrayInMemorySearch`)[1].

### Prompt & Retrieval Pipeline

The retrieval pipeline follows these steps:

1. User query is classified into categories (SQL, Wikipedia, GoogleSearch, VectorDBsearch, etc.) using an LLM[1].
2. Based on the classification, relevant data sources are queried:
   - SQL queries are generated and executed on the restaurant database[1].
   - Vector DB is searched for relevant PDF content[1].
   - Wikipedia and Google Search are queried for additional information[1].
3. Retrieved information is combined and used to generate a comprehensive response[1].

### LLM Integration & Prompt Engineering

The system uses OpenAI's GPT-3.5-turbo model for various tasks:

1. Query classification[1].
2. SQL query generation[1].
3. Final response generation[1].

Prompt engineering techniques:

1. System messages define the AI's role and capabilities[1].
2. Recent chat history is included for context in follow-up questions[1].
3. Prompts are structured to encourage concise and relevant responses[1].

### Factual Consistency & References

To ensure factual consistency:

1. The system combines information from multiple sources (database, PDF, Wikipedia, Google Search)[1].
2. Responses are generated based on retrieved facts rather than the LLM's general knowledge[1].
3. When information is unavailable or uncertain, the system acknowledges this in the response[1].

### Scalability & Future Extensions

To scale the system:

1. Replace the in-memory vector store with a scalable solution like Pinecone or FAISS.
2. Implement batch processing for large-scale data ingestion.
3. Add a caching layer for frequently accessed data and embeddings.
4. Implement incremental updates for the restaurant database and external sources.

Future improvements:

1. Real-time web scraping for up-to-date restaurant information.
2. User feedback mechanism to improve response quality.
3. Multi-language support for international users.

### Sample Queries

1. "What are the top-rated Italian restaurants in San Francisco?"
2. "Tell me about the history of sushi."
3. "What are the trending ingredients in vegetarian dishes this year?"
4. "Compare the average prices of Mexican and Chinese restaurants in the city."

### Additional Features

1. **Chatbot Interface**: The system includes a user-friendly Streamlit-based chat interface[2].
2. **Chat History**: The frontend maintains a chat history for context and user reference[2].
3. **Clear Chat Functionality**: Users can clear the chat history with a button click[2].

This MenuData AI Chatbot demonstrates a comprehensive approach to building an intelligent food and restaurant assistant, combining structured data, unstructured text, and external knowledge sources to provide informative and context-aware responses.
