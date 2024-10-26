import os
from flask import Flask, request, render_template
import torch
import torchaudio
from transformers import pipeline, AutoProcessor
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import soundfile as sf
import numpy as np
import requests
from dotenv import load_dotenv
from functools import lru_cache
import wave

load_dotenv()
app = Flask(__name__)

# Load API keys from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
hf_api_key = os.getenv("HF_API_KEY")

# Initialize models only once and cache them
@lru_cache(maxsize=1)
def get_asr_pipeline():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Use smaller whisper model
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=device,
    )

@lru_cache(maxsize=1)
def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

def convert_mp3_to_wav(mp3_file):
    """Convert MP3 to WAV using basic wave operations"""
    # Read MP3 file using torchaudio (smaller than pydub)
    waveform, sample_rate = torchaudio.load(mp3_file.stream)
    
    # Convert to mono and resample to 16kHz
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Save as WAV
    wav_buffer = wave.open(mp3_file.stream, 'wb')
    wav_buffer.setnchannels(1)
    wav_buffer.setsampwidth(2)
    wav_buffer.setframerate(16000)
    wav_buffer.writeframes(waveform.numpy().tobytes())
    return wav_buffer

def transcribe_audio(audio):
    """Transcribe audio with optimized settings"""
    if audio.filename.endswith('.mp3'):
        audio = convert_mp3_to_wav(audio)
    
    # Load audio directly with soundfile (smaller than librosa)
    audio_data, sr = sf.read(audio)
    if sr != 16000:
        # Simple resampling using numpy
        audio_data = np.interp(
            np.linspace(0, len(audio_data), int(len(audio_data) * 16000/sr)),
            np.arange(len(audio_data)),
            audio_data
        )
    
    asr_pipeline = get_asr_pipeline()
    result = asr_pipeline(
        {"array": audio_data, "sampling_rate": 16000},
        batch_size=8
    )
    return result['text']

def get_pdf_text(pdf_file):
    """Extract text from PDF with minimal memory usage"""
    text = []
    pdf_reader = PdfReader(pdf_file, strict=False)
    
    for page in pdf_reader.pages:
        chunk = page.extract_text()
        if chunk:
            text.append(' '.join(chunk.split()))
    
    return ' '.join(text)

def process_text(raw_text, user_question):
    """Process text with minimal memory usage"""
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=20000,
        chunk_overlap=1000,
        length_function=len
    ).split_text(raw_text)
    
    embeddings = get_embeddings_model()
    # Use simple cosine similarity instead of FAISS
    query_embedding = embeddings.embed_query(user_question)
    
    # Simple vector similarity search
    chunk_embeddings = [embeddings.embed_query(chunk) for chunk in chunks]
    similarities = [
        np.dot(query_embedding, chunk_emb) / 
        (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb))
        for chunk_emb in chunk_embeddings
    ]
    
    # Get top 2 most similar chunks
    top_indices = np.argsort(similarities)[-2:]
    relevant_chunks = [chunks[i] for i in top_indices]
    
    prompt = PromptTemplate(
        template="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        input_variables=["context", "question"]
    )
    
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=google_api_key,
        max_output_tokens=1024
    )
    
    chain = load_qa_chain(model, prompt=prompt)
    response = chain(
        {"input_documents": [{"page_content": c} for c in relevant_chunks], 
         "question": user_question},
        return_only_outputs=True
    )
    
    return response["output_text"]

def process_audio(audio, question):
    transcription = transcribe_audio(audio)
    return process_text(transcription, question)

def process_pdfs(pdfs, question):
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
    
    response = requests.post(
        API_URL,
        headers=headers,
        json={
            "inputs": f"Question: {question}\nAnswer:",
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
    app.run(debug=False)