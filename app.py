import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import sqlite3
from datetime import datetime
import uuid
import threading
from pathlib import Path
import time

# --- Load all env variables ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configure Gemini
genai.configure(api_key=gemini_api_key)

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_STORE_PATH = "vector_store"
DOCUMENT_DIR = "uploaded_docs"
ADMIN_PASSWORD = "admin@123"

# --- Define the models ---
MODELS_TO_USE = {
    "Gemini 1.5 Pro": GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=gemini_api_key),
    "OpenAI GPT-4o": ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
}

# --- Initialize session state ---
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = set()

# Create directories if they don't exist
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENT_DIR, exist_ok=True)

# --- Database setup for feedback ---
conn = sqlite3.connect('feedback.db', check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    answer_id TEXT UNIQUE,
    question TEXT,
    model_name TEXT,
    answer TEXT,
    feedback INTEGER,
    duration REAL,
    timestamp TEXT
)
''')
conn.commit()

def store_feedback(answer_id, question, model_name, answer, feedback, duration):
    timestamp = datetime.now().isoformat()
    try:
        c.execute(
            "INSERT INTO feedback (answer_id, question, model_name, answer, feedback, duration, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (answer_id, question, model_name, answer, feedback, duration, timestamp)
        )
        conn.commit()
        st.session_state.feedback_given.add(answer_id)
        return True
    except sqlite3.IntegrityError:
        st.session_state.feedback_given.add(answer_id)
        return False

# --- Embeddings and Document Processing ---
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return genai.embed_content(model='models/text-embedding-004', content=texts, task_type="retrieval_document")['embedding']
    def embed_query(self, text):
        return genai.embed_content(model='models/text-embedding-004', content=text, task_type="retrieval_query")['embedding']

embeddings = GeminiEmbeddings()

def process_documents():
    st.info("🔍 Loading documents...")
    documents = []
    for filename in os.listdir(DOCUMENT_DIR):
        file_path = os.path.join(DOCUMENT_DIR, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.endswith(".txt"):
                loader = TextLoader(file_path)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                continue
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {filename}: {str(e)}")
    if not documents:
        st.warning("No documents found to process.")
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    with st.spinner(f"Creating vector store from {len(texts)} text chunks..."):
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(VECTOR_STORE_PATH)
    st.success("✅ Documents processed and vector store created!")
    return vectorstore

def get_answers_from_all_models(query, vectorstore):
    responses = {}
    threads = []

    prompt_template = """
    You are a helpful university IT support assistant.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Format your answer for clear readability. Use paragraphs for explanations and bullet points for lists or steps.
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    def run_chain(model_name, llm):
        try:
            chain_type_kwargs = {"prompt": PROMPT}
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                return_source_documents=True,
                chain_type_kwargs=chain_type_kwargs
            )
            
            start_time = time.time()
            result = qa_chain.invoke(query)
            end_time = time.time()
            
            result['duration'] = end_time - start_time
            responses[model_name] = result

        except Exception as e:
            responses[model_name] = {"result": f"Error: {str(e)}", "source_documents": [], "duration": 0}

    for name, model_instance in MODELS_TO_USE.items():
        thread = threading.Thread(target=run_chain, args=(name, model_instance))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
        
    return responses

# --- Page Rendering Functions ---

def home_page():
    with st.container():
        st.markdown(
            """
            <div style="text-align: center; margin-top: 5rem;">
                <h1 style="font-size: 3rem; font-weight: bold;">Department of IT Chatbot</h1>
                <p style="font-size: 1.2rem; color: #888; margin-top: 1rem; margin-bottom: 2rem;">
                    Welcome! Ask a question to get answers from multiple AI models. 
                    Your feedback helps improve the system.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([2, 1.5, 2])
        with col2:
            if st.button("Start Chatbot", use_container_width=True, type="primary"):
                st.session_state.page = "chat"
                st.rerun()

def chat_page():
    st.title("💬 Department of IT Chatbot")

    with st.sidebar:
        if st.session_state.get('authenticated'):
            st.success("Admin Logged In")
            if st.button("📄 Document Management"):
                st.session_state.page = "admin"
                st.rerun()
            if st.button("📊 Feedback Dashboard"):
                st.session_state.page = "feedback"
                st.rerun()
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.rerun()
        else:
            st.subheader("Admin Login")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if password == ADMIN_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password")
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.feedback_given = set()
            st.toast("Chat history cleared!")
            st.rerun()

    if prompt := st.chat_input("Ask your IT question here..."):
        with st.spinner("🧠 Generating answers from all models... Please wait."):
            try:
                vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
                responses = get_answers_from_all_models(prompt, vectorstore)
                st.session_state.chat_history.append({"query": prompt, "responses": responses, "final_answer": None})
            except Exception as e:
                st.error(f"Could not load vector store. Please process documents in the admin panel. Error: {e}")

    if st.session_state.chat_history:
        for i, conversation in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(f"**Your Question:** {conversation['query']}")

            final_answer = conversation.get("final_answer")

            if final_answer:
                with st.chat_message("assistant"):
                    st.subheader(f"Your Selected Answer from {final_answer['model_name']}")
                    st.markdown(final_answer['response_text'])
                    with st.expander("Show Information Source"):
                        source_docs = final_answer.get('source_docs', [])
                        if source_docs:
                            unique_sources = set(Path(doc.metadata.get('source', 'Unknown')).name for doc in source_docs)
                            for source_name in unique_sources:
                                st.info(f"📄 {source_name}")
                        else:
                            st.warning("No source document information available.")
            else:
                with st.chat_message("assistant"):
                    st.markdown("#### Model Responses (Like the best one)")
                    
                    def handle_like(query_index, model_name):
                        convo = st.session_state.chat_history[query_index]
                        response_data = convo['responses'][model_name]
                        response_text = response_data.get('result', '')
                        duration = response_data.get('duration', 0)
                        answer_id = f"{convo['query']}-{model_name}".replace(" ", "_")
                        
                        if store_feedback(answer_id, convo['query'], model_name, response_text, 1, duration):
                            convo['final_answer'] = {'model_name': model_name, 'response_text': response_text, 'source_docs': response_data.get('source_documents', [])}
                        else:
                            convo['final_answer'] = {'model_name': model_name, 'response_text': response_text, 'source_docs': response_data.get('source_documents', [])}
                    
                    for model_name, response in conversation["responses"].items():
                        duration = response.get('duration', 0)
                        st.subheader(f"{model_name} `(Generated in {duration:.2f}s)`")
                        response_text = response.get('result', '')
                        st.markdown(response_text)
                        
                        answer_id = f"{conversation['query']}-{model_name}".replace(" ", "_")
                        is_disabled = answer_id in st.session_state.feedback_given

                        feedback_col1, feedback_col2, _ = st.columns([1, 1, 4])
                        with feedback_col1:
                            st.button("👍 Like", key=f"like_{i}_{model_name}", on_click=handle_like, args=(i, model_name), use_container_width=True, disabled=is_disabled)
                        with feedback_col2:
                            if st.button("👎 Dislike", key=f"dislike_{i}_{model_name}", use_container_width=True, disabled=is_disabled):
                                if store_feedback(answer_id, conversation['query'], model_name, response_text, 0, duration):
                                    st.toast(f"Feedback for {model_name} recorded.")
                                    st.rerun()
                                else:
                                    st.warning("Feedback already given for this answer.")
                        st.divider()

            st.markdown("---")

#  Admin page 
def admin_page():
    st.title("📄 Document Management")
    
    st.subheader("Upload Local Documents")
    uploaded_files = st.file_uploader("Choose files", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join(DOCUMENT_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved {uploaded_file.name}")
            
    st.markdown("---")
    st.subheader("Process Knowledge Base")
    st.info("Click here to add all new documents to the chatbot's memory.")
    if st.button("Process All New Documents", type="primary"):
        process_documents()
        
    st.markdown("---")
    st.subheader("Available Documents")
    files = os.listdir(DOCUMENT_DIR)
    if not files:
        st.warning("No documents available.")
    else:
        for filename in files:
            col1, col2 = st.columns([0.8, 0.2])
            col1.write(f"📝 {filename}")
            if col2.button("Delete", key=f"del_{filename}"):
                os.remove(os.path.join(DOCUMENT_DIR, filename))
                st.success(f"Deleted {filename}")
                st.rerun()
                
    if st.button("⬅️ Back to Chat"):
        st.session_state.page = "chat"
        st.rerun()


def admin_feedback_page():
    st.title("📊 Feedback Dashboard")
    
    with st.expander("Advanced Options"):
        st.warning("This action will permanently delete all feedback data from the database.")
        if st.button("🗑️ Clear All Feedback Data", type="primary"):
            c.execute("DELETE FROM feedback")
            conn.commit()
            st.success("All feedback data has been cleared.")
            st.rerun()
            
    st.markdown("---")
    st.header("Overall Performance")
    c.execute("SELECT COUNT(DISTINCT question) FROM feedback")
    total_questions = c.fetchone()[0] or 0
    c.execute("SELECT COUNT(*) FROM feedback WHERE feedback=1")
    total_likes = c.fetchone()[0] or 0
    c.execute("SELECT COUNT(*) FROM feedback WHERE feedback=0")
    total_dislikes = c.fetchone()[0] or 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Questions with Feedback", total_questions)
    col2.metric("Total 👍 Likes", total_likes)
    col3.metric("Total 👎 Dislikes", total_dislikes)

    st.markdown("---")
    st.header("Model Performance Breakdown")
    
    c.execute("SELECT model_name, SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END), SUM(CASE WHEN feedback = 0 THEN 1 ELSE 0 END), AVG(duration) FROM feedback GROUP BY model_name")
    model_stats = c.fetchall()
    
    if not model_stats:
        st.info("No feedback has been recorded yet to display model performance.")
    else:
        num_models = len(MODELS_TO_USE)
        cols = st.columns(num_models if num_models > 0 else 1)
        
        stats_map = {name: (likes, dislikes, avg_duration) for name, likes, dislikes, avg_duration in model_stats}

        for i, model_name in enumerate(MODELS_TO_USE.keys()):
            with cols[i]:
                st.subheader(model_name)
                if model_name in stats_map:
                    likes, dislikes, avg_duration = stats_map[model_name]
                    st.metric(label="👍 Likes", value=int(likes or 0))
                    st.metric(label="👎 Dislikes", value=int(dislikes or 0))
                    st.metric(label="Avg. Response Time", value=f"{avg_duration or 0:.2f}s")
                else:
                    st.metric(label="👍 Likes", value=0)
                    st.metric(label="👎 Dislikes", value=0)
                    st.metric(label="Avg. Response Time", value="N/A")

    st.markdown("---")
    st.header("Detailed Feedback Log")
    c.execute("SELECT question, model_name, answer, feedback, timestamp FROM feedback ORDER BY timestamp DESC LIMIT 50")
    rows = c.fetchall()

    for question, model, answer, feedback, ts in rows:
        icon = "👍" if feedback == 1 else "👎"
        with st.expander(f"{icon} **{model}** | Q: *{question[:60]}...*"):
            st.text(f"Timestamp: {datetime.fromisoformat(ts).strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**Question:** {question}")
            st.markdown(f"**Answer:** {answer}")

    if st.button("⬅️ Back to Chat"):
        st.session_state.page = "chat"
        st.rerun()

# --- Main App Routing Logic ---
if __name__ == "__main__":
    page = st.session_state.get('page', 'home')
    if page == "home":
        home_page()
    elif page == "chat":
        chat_page()
    elif page == "admin" and st.session_state.get('authenticated'):
        admin_page()
    elif page == "feedback" and st.session_state.get('authenticated'):
        admin_feedback_page()
    else:
        st.session_state.page = "home"
        st.rerun()