import streamlit as st
import requests
import json

# Set up page configuration
st.set_page_config(page_title='MediBOT - Medical Chat', page_icon='💬', layout='wide')

# Custom Styles
st.markdown("""
    <style>
        .title-container {
            text-align: center;
            margin-bottom: 20px;
        }
        .chat-history-container {
            text-align: left;
            margin-left: 245px;
            margin-bottom: 10px;
        }
        .chat-container {
            padding: 15px;
            border-radius: 10px;
            max-height: 500px;
            overflow-y: auto;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        .message-box {
            padding: 10px;
            border-radius: 10px;
            margin: 7px 250px;
            max-width: 38%;
            font-size: 18px;
            font-weight: normal;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #b2b5e0;
            color: black;
            text-align: left;
            margin-left: auto;
        }
        .assistant-message {
            background-color: #ddd0c8;
            color: black;
            text-align: left;
            margin-right: auto;
        }
        .chat-input-container {
            width: 75%;
            max-width: 600px;
            padding-top: 10px;
            text-align: center;
        }
        [data-testid="stChatInput"] {
            max-width: 1290px !important;
            width: 100% !important;
            margin: 0 auto;
            display: block;
            text-align: left;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# App Title
st.markdown("<div class='title-container'><h1>💬 MediBOT - Medical Chat</h1></div>", unsafe_allow_html=True)

# Chat History Display
st.markdown("<div class='chat-history-container'><h3>💬 Chat History</h3></div>", unsafe_allow_html=True)
with st.container():
    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        st.markdown(
            f"<div class='message-box {role_class}'>{message['content']}</div>",
            unsafe_allow_html=True
        )

# Chat Input Field
st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)
query = st.chat_input("✍️ Ask a medical question...")
st.markdown("</div>", unsafe_allow_html=True)

# Start New Conversation
if st.button("🚀 Start New Conversation"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.chat_id = None

# If there's a user input
if query:
    # Add user message to session state
    st.session_state.messages.append({'role': 'user', 'content': query})
    
    # API URL (Make sure FastAPI is running and accessible)
    url = "http://127.0.0.1:9595/qnaConversation"
    
    headers = {"Content-Type": "application/json"}
    body = {
        "chat_id": st.session_state.chat_id,
        "user_query": query,
        "Id": 1,  # Default Id for simplicity
        "patientId": "12345",  # Sample Patient ID
        "name": "John Doe",
        "age": 30,
        "gender": "Male",
        "occupation": "Engineer",
        "firstMessage": True,
        "language_code": "en"
    }
    
    try:
        # Send the request to FastAPI
        res = requests.post(url, headers=headers, json=body)
        if res.status_code == 200:
            data = res.json()
            st.session_state.chat_id = data["chat_id"]
            assistant_response = data["message"]
            st.session_state.messages.append({'role': 'assistant', 'content': assistant_response})
        else:
            st.session_state.messages.append({'role': 'assistant', 'content': "❌ Error: Unable to process request."})
    except Exception as ex:
        st.session_state.messages.append({'role': 'assistant', 'content': f"❌ Error: {ex}"})
    
    # Refresh the chat with the new message
    st.experimental_rerun()
