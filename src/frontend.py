import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/chat/"

# Streamlit UI
st.set_page_config(page_title="MenuData Chatbot", page_icon="🍽️")
# Load and display the company logo in the header
logo_path = "../images/menudataai_logo.jpeg"  # Replace with the correct path to your logo file

col1, col2 = st.columns([1, 5])  # Adjust column width for alignment

with col1:
    st.image(logo_path, width=90)  # Display logo with a fixed width

with col2:
    st.markdown("## MenuData Chatbot 🍽️")  # Chatbot title
    st.markdown("Ask about restaurants, menus, ingredients, or food history!")

st.markdown("---")  # Adds a horizontal line for separation

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
query = st.chat_input("Type your message...")

if query:
    # Display user message
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Call FastAPI chatbot
    response = requests.post(API_URL, json={"query": query})
    if response.status_code == 200:
        bot_response = response.json()["response"]
    else:
        bot_response = "Sorry, I couldn't fetch a response."

    # Display bot response
    st.session_state["messages"].append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.write(bot_response)

# Clear chat button
if st.button("Clear Chat 🗑️"):
    st.session_state["messages"] = []
    st.rerun()
