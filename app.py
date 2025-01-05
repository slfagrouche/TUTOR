import os
import io
import re
import requests
# import torch
import librosa

from pydub import AudioSegment
from flask import Flask, request, render_template, redirect, url_for, session
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

# # Initialize and configure the Whisper and PDF processing tools
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# # Load Whisper model
# model_id = "openai/whisper-large-v3"
# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id, 
#     torch_dtype=torch_dtype, 
#     use_safetensors=True
# )
# model.to(device)
# processor = AutoProcessor.from_pretrained(model_id)

# # Define the ASR pipeline
# asr_pipeline = pipeline(
#     "automatic-speech-recognition", 
#     model=model, 
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor, 
#     device=device
# )

# Middleware to ensure Google API Key is set
@app.before_request
def require_api_key():
    """
    Ensure a Google API Key is set in the session before accessing routes.
    """
    if request.endpoint not in ['setup', 'static']:
        if 'google_api_key' not in session:
            return redirect(url_for('setup'))

# Setup route for Google API Key
@app.route('/setup', methods=['GET', 'POST'])
def setup():
    """
    Setup page to collect and validate the user's Google API Key.
    """
    if request.method == 'POST':
        google_api_key = request.form.get('google_api_key')
        if google_api_key:
            if validate_google_api_key(google_api_key):
                session['google_api_key'] = google_api_key
                return redirect(url_for('index'))
            else:
                return render_template('setup.html', error="Invalid Google API key. Please try again.")
        else:
            return render_template('setup.html', error="Please enter a valid Google API key.")
    return render_template('setup.html')

def validate_google_api_key(api_key):
    """
    Validate the provided Google API Key using the Generative Language API.
    
    Args:
        api_key (str): Google API Key to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"role": "user", "parts": [{"text": "Test validation"}]}],
                "generationConfig": {"responseMimeType": "application/json"}
            },
            timeout=5  # Faster timeout
        )
        if response.ok:
            app.logger.info("✅ API Key is valid.")
            return True
        app.logger.error(f"❌ Invalid API Key. Status Code: {response.status_code}. Response: {response.text}")
        return False
    except requests.exceptions.RequestException as e:
        app.logger.error(f"❌ Network or Request Error: {e}")
        return False


# Audio Processing Functions
def convert_mp3_to_wav(mp3_audio):
    mp3_path = mp3_audio.stream
    mp3_sound = AudioSegment.from_file(mp3_path, format="mp3")
    buffer = io.BytesIO()
    mp3_sound.export(buffer, format="wav")
    buffer.seek(0)
    return buffer

def transcribe_audio(audio):
    if audio.filename.endswith('.mp3'):
        audio = convert_mp3_to_wav(audio)
    
    audio_data, sr = librosa.load(audio, sr=16000)
    result = asr_pipeline({"array": audio_data, "sampling_rate": sr})
    return result['text']

# PDF Processing Function
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Question Answering
def answer_question(user_question, pdf_text=None, audio_text=None):
    raw_text = (pdf_text or "") + (audio_text or "")
    if raw_text == "":
        return "No content to process. Please upload a PDF or audio file."

    text_chunks = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000).split_text(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=session.get('google_api_key')
    )
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
    llm_model = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        temperature=0.5, 
        google_api_key=session.get('google_api_key')
    )
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(llm_model, prompt=prompt)

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/audio', methods=['GET', 'POST'])
def audio():
    if request.method == 'POST':
        audio_file = request.files['audio']
        question = request.form.get('question')
        # response = process_audio(audio_file, question)
        response = "Error processing audio :( Sorry about that, Said is working on it!"
        return render_template('audio.html', response=response)
    return render_template('audio.html', response=None)

def process_audio(audio_file, question):
    audio_text = transcribe_audio(audio_file)
    return answer_question(question, None, audio_text)

@app.route('/pdf', methods=['GET', 'POST'])
def pdf():
    """
    Handle PDF file uploads and process them to answer a user's question.
    """
    if request.method == 'POST':
        pdfs = request.files.getlist('pdf')
        question = request.form.get('question')
        response = process_pdfs(pdfs, question)
        return render_template('pdf.html', response=response)
    return render_template('pdf.html', response=None)

@app.route('/general', methods=['GET', 'POST'])
def general():
    if request.method == 'POST':
        question = request.form.get('question')
        response = answer_general_question(question)
        return render_template('general.html', response=response)
    return render_template('general.html', response=None)

def answer_general_question(user_question):
    """
    A fallback function using a Hugging Face model via your HF_API_KEY.
    """
    API_URL = "https://api-inference.huggingface.co/models/google/gemma-1.1-7b-it"
    headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

    payload = {
        "inputs": f"{user_question}\nPlease format the response in clean Markdown, including clear sections and headings.",
        "parameters": {"max_new_tokens": 1000, "return_full_text": False}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        output = response.json()
        if isinstance(output, list) and 'generated_text' in output[0]:
            raw_text = output[0]['generated_text']
            # Remove repeated user question from the output
            clean_text = re.sub(re.escape(user_question), '', raw_text, flags=re.IGNORECASE).strip()
            return clean_text
        return "Unexpected API response. Please try again later."
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

def process_pdfs(pdfs, question):
    """
    Process PDF files and answer a question based on their content.
    """
    extracted_text = ""
    for pdf in pdfs:
        extracted_text += get_pdf_text(pdf)
    return answer_question(question, extracted_text, None)

if __name__ == '__main__':
    # Use debug=False in production
    app.run(debug=True)
