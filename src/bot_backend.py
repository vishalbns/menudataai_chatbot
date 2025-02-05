import os
import wikipedia
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper
from requests import request
import geocoder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
_ = load_dotenv(find_dotenv())  # Read local .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATABASE_URL = os.environ["DATABASE_URL"]
SERPAPI_KEY = os.environ["SERPAPI_KEY"]  # For Google search

# Initialize FastAPI
app = FastAPI()

# Database connection
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

# Input model
class QueryRequest(BaseModel):
    query: str
    location: Optional[str] = None  # For queries like "near me"

# Global list to store chat history as a list of dicts {"user": ..., "bot": ...}
global chat_history
chat_history = []

# Helper function: return recent chat history as a string.
# It takes the last two Q/A pairs and trims the string if it exceeds max_length.
def get_recent_history(max_entries=2, max_length=250) -> str:
    recent_entries = chat_history[-max_entries:]
    history_str = "\n".join([f"User: {entry['user']}\nBot: {entry['bot']}" for entry in recent_entries])
    if len(history_str) > max_length:
        history_str = history_str[-max_length:]
    return history_str

# Update chat history and trim older interactions after 10 entries
def update_chat_history(user_query: str, bot_response: str):
    global chat_history
    # If a greeting is detected, assume a new conversation and clear history
    if classify_query_with_llm(user_query) == "Greeting":
        chat_history = []
    else:
        chat_history.append({"user": user_query, "bot": bot_response})
        # Archive older history by keeping only the last 10 interactions
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

def load_and_store_pdf():
    """Loads the predefined PDF, extracts text, and stores embeddings in vector DB."""
    global vector_db

    pdf_path = "../internal_docs/evolution_of_american_food.pdf"  # Ensure this file is in your project directory
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Generate embeddings using OpenAI
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Store in in-memory vector database
    vector_db = DocArrayInMemorySearch.from_documents(docs, embeddings)

# Load the PDF on startup
load_and_store_pdf()


# Function to generate SQL query using LLM
def generate_sql_query(user_query: str):
    prompt = f"""
    You are an AI trained to generate SQL queries for a PostgreSQL database based on user input.
    The database has a table named `restaurant_menu` with the following columns:
    - `restaurant_name` (VARCHAR)
    - `menu_category` (VARCHAR)
    - `item_id` (INTEGER)
    - `menu_item` (VARCHAR)
    - `menu_description` (TEXT)
    - `ingredient_name` (VARCHAR)
    - `confidence` (FLOAT)
    - `categories` (VARCHAR)
    - `address1` (TEXT)
    - `city` (VARCHAR)
    - `zip_code` (VARCHAR)
    - `country` (VARCHAR)
    - `state` (VARCHAR)
    - `rating` (VARCHAR)
    - `review_count` (VARCHAR)
    - `price` (VARCHAR)

    Convert the following user query into a valid SQL query for the `restaurant_menu` table:
    "{user_query}"

    Only return the SQL query, no extra explanations, and do not include any code formatting markers.
    Remember review_count, rating, and price columns are varchar. Perform cast conversion where necessary.
    Don't forget to use DISTINCT as there are many duplicate restaurant names, ingredients, menu items etc.
    Everything in the database is lower case, so remember to lowercase query values.
    """

    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=100)

    return response.choices[0].message.content.strip()

# Function to execute SQL query
def execute_sql_query(sql: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        if not results:
            return "No matching results found."
        formatted_results = "\n".join([", ".join(map(str, row)) for row in results])
        return formatted_results
    except Exception as e:
        return f"Error executing query: {e}"
    finally:
        cursor.close()
        conn.close()

# Function to query restaurant database
def query_restaurant_db(query: str):
    sql_query = generate_sql_query(query)
    return execute_sql_query(sql_query)

# Wikipedia search function
def search_wikipedia(query: str):
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.PageError:
        return "No relevant Wikipedia article found."
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Did you mean: {', '.join(e.options[:5])}?"

# Google Search using SerpAPI
def search_google(query: str):
    google_search = SerpAPIWrapper(serpapi_api_key=SERPAPI_KEY)
    return google_search.run(query)


def vector_db_search(query: str):
    """Retrieves relevant sections from the stored PDF based on user query."""
    if vector_db is None:
        return {"error": "VectorDB not initialized."}

    retriever = vector_db.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo")  # Using GPT-3.5 Turbo for responses
    chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    response = chain.run(query)
    return response

# LLM response function
def generate_response(prompt: str):
    prompt = f""" 
    Generate response for the prompt: {prompt}
    You can refer to the chat history to provide answers only if the latest prompt is a follow up. 
    Do not apologise and do not mention any confusion unnecessarily.
    Give a crisp, friendly, and polite response. If you are not sure, say you are not sure and advise reaching out to customer care.
    """
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful food assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=512)
    return response.choices[0].message.content

# Helper function to classify query using LLM (with recent chat history included)
def classify_query_with_llm(query: str) -> list:
    # Include a brief recent history to help with follow-ups
    prompt = f"""
    New user query: "{query}"
    
    Classify the query into one or more of the following categories: Greeting, SQL, Wikipedia, GoogleSearch, Irrelevant.
    I have a database that has restaurant data with names of the restaurants, menu items, ingredients and addresses of the restaurant in San Francisco.
    Categories:
    - Greeting: Greetings like "Hello", "Hi", "Hey", "How are you?", "Good morning", "Good evening", "What can you do".
    - VectorDBsearch: Query is about evolution of american food, thanksgiving, etc.
    - SQL: Queries that involve database searches, like finding restaurants, prices, or ratings.
    - Wikipedia: Queries related to food items like sushi, food ingredients, cuisine history, nutrition, or similar background information.
    - GoogleSearch: Queries asking for trending, popular, or recent food-related information, such as reviews or new restaurants.
    - Irrelevant: If the query is about politics, violence, technology, sports, general knowledge, or anything unrelated to food or restaurants.
    
    Return a comma-separated list. Ex: for query "What is the history of sushi, and which restaurants in my area are known for it?", it should return "Wikipedia", "GoogleSearch"
    """
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=20)
    categories = response.choices[0].message.content.strip()
    return [cat.strip() for cat in categories.split(",") if cat.strip()]

def is_follow_up_query(query, recent_history):
    """
    Determines whether a given query is a follow-up to the recent chat history using an LLM.
    """
    if not recent_history:
        return False  # No history means no follow-up

    follow_up_prompt = f"""
    Chat history:
    {recent_history}

    New user query: "{query}"

    Determine if the new query is a follow-up to the previous conversation.
    If new query has phrases like "list them", "give me its recipe", etc. that talk about things in previous conversation, say "Yes",
    If the new query seems independent and can do without previous conversation, say "No".
    Respond with only one word: "Yes" or "No".
    """

    follow_up_response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are an intelligent assistant."},
        {"role": "user", "content": follow_up_prompt}
    ],
    max_tokens=5)

    return follow_up_response.choices[0].message.content.strip().lower() == "yes"



@app.post("/chat/")
def chatbot(request: QueryRequest):

    query = request.query.lower()
    recent_history = get_recent_history(max_entries=2, max_length=256)
    is_follow_up = is_follow_up_query(query, recent_history)
    combined_query = f"{recent_history} {query}" if is_follow_up else query

    query_types = classify_query_with_llm(combined_query)
    print(is_follow_up)
    print(query_types)
    print(combined_query)
    location = geocoder.ip('me').address

    # If the query is a greeting, clear history (new conversation) and respond without context.
    if "Greeting" in query_types:
        chat_history.clear()
        bot_response = "Hello! I’m your restaurant and food assistant. You can ask me about restaurants, menus, food history, or reviews!"
    else:
        # Get recent history (if any) to include in the prompt for follow-up questions.
        recent_history = get_recent_history(max_entries=2, max_length=256)
        is_follow_up = is_follow_up_query(query, recent_history)
        combined_query = f"{recent_history} {query}" if is_follow_up else query

        query_types = classify_query_with_llm(combined_query)  # returns a list
        bot_responses = []
        
        if "VectorDBsearch" in query_types:
            bot_responses.append(vector_db_search(query))
        
        if "SQL" in query_types:
            db_response = query_restaurant_db(combined_query)
            full_prompt = (f"Chat history:\n{recent_history}\nUser asked: {query}.\nDatabase says:\n{db_response}") if is_follow_up else (f"User asked: {query}.\nDatabase says:\n{db_response}")
            bot_responses.append(generate_response(full_prompt))

        if "Wikipedia" in query_types:
            wiki_query = combined_query[-300:] if len(combined_query) > 300 else combined_query
            wiki_info = search_wikipedia(wiki_query)
            full_prompt = (f"Chat history:\n{recent_history}\nUser asked: {query}.\nWikipedia info:\n{wiki_info}") if is_follow_up else (f"User asked: {query}.\nWikipedia info:\n{wiki_info}")
            bot_responses.append(generate_response(full_prompt))

        if "GoogleSearch" in query_types:
            web_query = combined_query[-300:] if len(combined_query) > 300 else combined_query
            if "near me" or "my area" in web_query.lower() and location:
                web_query += f" in {location}"
            web_info = search_google(web_query)
            full_prompt = (f"Chat history:\n{recent_history}\nUser asked: {query}.\nWeb search results:\n{web_info}") if is_follow_up else (f"User asked: {query}.\nWeb search results:\n{web_info}")
            bot_responses.append(generate_response(full_prompt))

        # Generate AI response based on combined results
        if bot_responses:
            full_prompt = f"User asked: {query}\n\n" + "\n\n".join(bot_responses)
            bot_response = generate_response(full_prompt)
        else:
            bot_response = "I can only answer food-related or restaurant queries."

        update_chat_history(query, bot_response)

    return {"response": bot_response}

# Run the API on localhost
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
