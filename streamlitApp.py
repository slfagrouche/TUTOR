import os
import io
import requests
import streamlit as st
import librosa
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Initialize Session State for Navigation ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def navigate_to(page):
    """Navigate between pages explicitly"""
    st.session_state.page = page
    st.rerun()

# --- App Setup ---
st.set_page_config(
    page_title="AI Tutor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utility Functions ---
def validate_google_api_key(api_key):
    """Validate Google API Key"""
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"role": "user", "parts": [{"text": "Test validation"}]}],
                "generationConfig": {"responseMimeType": "application/json"}
            },
            timeout=5
        )
        return response.ok
    except requests.exceptions.RequestException as e:
        print(f"❌ Network or Request Error: {e}")
        return False


def get_pdf_text(pdf_file):
    """Extract text from PDF"""
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def transcribe_audio(audio):
    """Transcribe audio file"""
    audio_data, sr = librosa.load(io.BytesIO(audio.read()), sr=16000)
    return "Audio transcription feature under development."


def answer_question(user_question, pdf_text=None, audio_text=None):
    """Answer a question based on PDF or Audio text"""
    raw_text = (pdf_text or "") + (audio_text or "")
    if raw_text == "":
        return "No content to process. Please upload a PDF or audio file."

    text_chunks = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000).split_text(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    docs = vector_store.similarity_search(user_question)

    prompt_template = """
    Based on the educational material provided—answer the student's question in detail.

    Uploaded Educational Material:
    {context}

    Student's Question:
    {question}

    Tutor's Response:
    """
    llm_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5, google_api_key=os.getenv("GOOGLE_API_KEY"))
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm_model, prompt=prompt)

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


# --- Sidebar for API Key ---
with st.sidebar:
    st.markdown("## 🔑 **Setup API Key**")
    st.write("Enter your **Google API Key** to unlock all features of the AI Tutor.")
    api_key = st.text_input("Enter your Google API Key", type="password")
    if st.button("Validate API Key"):
        if validate_google_api_key(api_key):
            st.success("✅ API Key is valid!")
            os.environ["GOOGLE_API_KEY"] = api_key
        else:
            st.error("❌ Invalid API Key. Please try again.")
    st.write("---")
    st.markdown("**Navigate:**")
    if st.button("🏠 Home"):
        navigate_to("Home")
    if st.button("📄 Upload PDF"):
        navigate_to("Upload PDF")
    if st.button("🎙️ Upload Audio"):
        navigate_to("Upload Audio")
    if st.button("💬 Ask Questions"):
        navigate_to("Ask General Questions")


# --- Navigation Logic ---
# --- Home Page with Improved Button Layout ---
if st.session_state.page == "Home":
    st.markdown("""
    <style>
    .hero-section {
        text-align: center; 
        background: linear-gradient(to right, #4F46E5, #9333EA);
        border-radius: 10px;
        padding: 40px;
        color: white;
    }
    .description-section {
        background: #1E1E2E;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        color: #FFFFFF;
    }
    .feature-box {
        border: 1px solid #44475A;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        background-color: #2A2A3C;
        color: #FFFFFF;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .feature-box h4 {
        margin: 10px 0;
        font-size: 18px;
        font-weight: bold;
    }
    .feature-box p {
        font-size: 14px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class='hero-section'>
        <h1>🤖 Welcome to TUTOR!</h1>
        <p>Your AI-powered educational assistant for smarter learning and seamless content interaction.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Description Section
    st.markdown("""
    <div class='description-section'>
        <h3>📚 What is TUTOR?</h3>
        <p>
            <strong>TUTOR</strong> is an AI-powered educational assistant designed to help users with
            <strong>audio transcription</strong>, <strong>PDF text extraction</strong>, and <strong>question answering</strong>.
            It leverages advanced AI models and <strong>Retrieval-Augmented Generation (RAG)</strong>
            to provide accurate and contextually relevant responses based on uploaded content.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Section
    st.write("### 🚀 **Key Features**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='feature-box'>
            <h4>🎙️ Audio Transcription</h4>
            <p>Upload audio files and get accurate transcriptions instantly.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-box'>
            <h4>📄 PDF Text Extraction</h4>
            <p>Extract valuable insights from PDF documents effortlessly.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-box'>
            <h4>💬 Question Answering</h4>
            <p>Ask questions based on uploaded content and get AI-powered answers.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='feature-box'>
            <h4>🧠 General AI Tutor</h4>
            <p>Ask general questions and receive context-aware AI responses.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Pitch Section
    st.write("### 🎯 **Why TUTOR is Different?**")
    st.markdown("""
    <div class='description-section'>
        <p>
            Traditional LLM applications often struggle with delivering accurate, context-aware answers.
            <strong>TUTOR</strong> leverages <strong>Retrieval-Augmented Generation (RAG)</strong>
            to combine the power of large language models with a curated knowledge base.
        </p>
        <p>
            This ensures responses are not just generated based on training data but are
            <strong>grounded in the user's uploaded content</strong>—be it audio transcriptions or PDF text.
            This significantly enhances the accuracy and relevance of the information provided.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Action Buttons
    st.write("---")
    st.write("### 🛠️ **Get Started**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💬 Ask a Question", key="ask_question"):
            navigate_to("Ask General Questions")
    with col2:
        if st.button("🎙️ Upload Audio", key="upload_audio"):
            navigate_to("Upload Audio")
    with col3:
        if st.button("📄 Upload PDF", key="upload_pdf"):
            navigate_to("Upload PDF")

elif st.session_state.page == "Upload PDF":
    st.header("📄 Upload PDFs")
    uploaded_pdfs = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
    question = st.text_input("Ask a question about the PDFs:")
    if st.button("Submit PDF Question"):
        if uploaded_pdfs and question:
            raw_text = "".join([get_pdf_text(pdf) for pdf in uploaded_pdfs])
            response = answer_question(question, raw_text, None)
            st.write("### 📚 Response:")
            st.markdown(response)
    if st.button("Back to Home"):
        navigate_to("Home")

elif st.session_state.page == "Upload Audio":
    st.header("🎙️ Upload Audio")
    uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
    question = st.text_input("Ask a question about the audio:")
    if st.button("Submit Audio Question"):
        audio_text = transcribe_audio(uploaded_audio)
        response = answer_question(question, None, audio_text)
        st.write("### 🎤 Response:")
        st.markdown(response)

elif st.session_state.page == "Ask General Questions":
    st.header("💬 Ask a General Question")
    general_question = st.text_input("Enter your question:")
    if st.button("Submit Question"):
        response = answer_question(general_question)
        st.write("### 🤖 Response:")
        st.markdown(response)

# --- Footer ---
st.markdown("""
<hr>
<div style='text-align: center; font-size: 14px;'>
    <p>Built with ❤️ using <b>Streamlit</b>, <b>LangChain</b>, and <b>Google Generative AI</b>.</p>
    <p>🛠️ <b>Developed by Said Lfagrouche</b.</p>
</div>
""", unsafe_allow_html=True)
