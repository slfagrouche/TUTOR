import os
from pydub import AudioSegment
import io
from flask import Flask, request, render_template
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import librosa
import requests
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()
app = Flask(__name__)

# Load API keys from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
hf_api_key = os.getenv("HF_API_KEY")

# Initialize models only once and cache them
@lru_cache(maxsize=1)
def get_asr_pipeline():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch_dtype,
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        chunk_length_s=30  # Process audio in chunks
    )

@lru_cache(maxsize=1)
def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

def convert_mp3_to_wav(mp3_audio):
    """Convert MP3 to WAV with reduced quality for smaller size"""
    mp3_sound = AudioSegment.from_file(mp3_audio.stream, format="mp3")
    # Reduce to mono and lower sample rate
    mp3_sound = mp3_sound.set_channels(1).set_frame_rate(16000)
    buffer = io.BytesIO()
    mp3_sound.export(buffer, format="wav", parameters=["-q:a", "0"])
    buffer.seek(0)
    return buffer

def transcribe_audio(audio):
    """Transcribe audio with optimized settings"""
    if audio.filename.endswith('.mp3'):
        audio = convert_mp3_to_wav(audio)
    
    # Load audio with reduced quality
    audio_data, sr = librosa.load(audio, sr=16000, mono=True)
    
    # Process in chunks to reduce memory usage
    asr_pipeline = get_asr_pipeline()
    result = asr_pipeline(
        {"array": audio_data, "sampling_rate": sr},
        batch_size=8,
        return_timestamps=False  # Disable if not needed
    )
    return result['text']

def get_pdf_text(pdf_file):
    """Extract text from PDF with optimized memory usage"""
    text_chunks = []
    pdf_reader = PdfReader(pdf_file, strict=False)
    
    for page in pdf_reader.pages:
        chunk = page.extract_text()
        if chunk:
            # Basic cleaning to reduce noise
            chunk = ' '.join(chunk.split())
            text_chunks.append(chunk)
    
    return ' '.join(text_chunks)

def process_text(raw_text, user_question):
    """Process text with optimized chunking and embedding"""
    # Use larger chunk size to reduce number of embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=20000,
        chunk_overlap=1000,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    
    # Create and search vector store
    vector_store = FAISS.from_texts(
        chunks,
        embedding=get_embeddings_model()
    )
    
    # Limit number of returned documents
    docs = vector_store.similarity_search(user_question, k=2)
    
    # Use simplified prompt template
    prompt = PromptTemplate(
        template="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        input_variables=["context", "question"]
    )
    
    # Initialize model with lower temperature
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=google_api_key,
        max_output_tokens=1024  # Limit response length
    )
    
    chain = load_qa_chain(model, prompt=prompt)
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    return response["output_text"]

def process_audio(audio, question):
    """Process audio files with optimized memory usage"""
    transcription = transcribe_audio(audio)
    return process_text(transcription, question)

def process_pdfs(pdfs, question):
    """Process PDFs with optimized memory usage"""
    extracted_text = ' '.join(get_pdf_text(pdf) for pdf in pdfs)
    return process_text(extracted_text, question)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/audio', methods=['GET', 'POST'])
def audio():
    if request.method == 'POST':
        audio = request.files['audio']
        question = request.form.get('question')
        response = process_audio(audio, question)
        return render_template('audio.html', response=response)
    return render_template('audio.html', response=None)

@app.route('/pdf', methods=['GET', 'POST'])
def pdf():
    if request.method == 'POST':
        pdfs = request.files.getlist('pdf')
        question = request.form.get('question')
        response = process_pdfs(pdfs, question)
        return render_template('pdf.html', response=response)
    return render_template('pdf.html', response=None)

@lru_cache(maxsize=100)
def get_general_response(question):
    """Cache general responses to reduce API calls"""
    API_URL = "https://api-inference.huggingface.co/models/google/gemma-1.1-7b-it"
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    
    prompt = f"Question: {question}\nAnswer:"
    
    response = requests.post(
        API_URL,
        headers=headers,
        json={
            "inputs": prompt,
            "parameters": {"max_length": 1024, "temperature": 0.3}
        }
    )
    return response.json()[0]['generated_text']

@app.route('/general', methods=['GET', 'POST'])
def general():
    if request.method == 'POST':
        question = request.form.get('question')
        response = get_general_response(question)
        return render_template('general.html', response=response)
    return render_template('general.html', response=None)

if __name__ == '__main__':
    app.run(debug=False)  # Disable debug mode in production