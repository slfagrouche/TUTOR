import os
from flask import Flask, request, render_template, session, jsonify, flash
import torch
import torchaudio
from transformers import pipeline
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import soundfile as sf
import numpy as np
import requests
from dotenv import load_dotenv
from functools import lru_cache
import wave
import secrets
import hashlib
from werkzeug.utils import secure_filename
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

# Configure file upload settings
UPLOAD_FOLDER = Path('uploads')
ALLOWED_EXTENSIONS_AUDIO = {'mp3', 'wav'}
ALLOWED_EXTENSIONS_PDF = {'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Global variables for API keys
google_api_key = os.getenv("GOOGLE_API_KEY")
hf_api_key = os.getenv("HF_API_KEY")

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def validate_api_key(api_key):
    """Validate API key format"""
    if not api_key:
        return False
    return bool(api_key.strip()) and api_key.replace('-', '').replace('_', '').isalnum()

def get_api_keys():
    """Get API keys from session or environment"""
    return google_api_key, hf_api_key

@lru_cache(maxsize=1)
def get_asr_pipeline():
    """Initialize ASR pipeline with caching"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=device,
    )

@lru_cache(maxsize=1)
def get_embeddings_model(api_key):
    """Initialize embeddings model with caching"""
    if not validate_api_key(api_key):
        raise ValueError("Invalid Google API key")
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

def convert_mp3_to_wav(mp3_file):
    """Convert MP3 to WAV format"""
    try:
        waveform, sample_rate = torchaudio.load(mp3_file)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        wav_path = UPLOAD_FOLDER / f"{secure_filename(mp3_file.filename)}.wav"
        torchaudio.save(wav_path, waveform, 16000)
        return wav_path
    except Exception as e:
        raise ValueError(f"Error converting audio file: {str(e)}")

def transcribe_audio(audio_file):
    """Transcribe audio file to text"""
    try:
        if audio_file.filename.endswith('.mp3'):
            wav_path = convert_mp3_to_wav(audio_file)
            audio_data, sr = sf.read(wav_path)
            wav_path.unlink()  # Clean up temporary file
        else:
            audio_data, sr = sf.read(audio_file)
        
        if sr != 16000:
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
    except Exception as e:
        raise ValueError(f"Error transcribing audio: {str(e)}")

def get_pdf_text(pdf_file):
    """Extract text from PDF file"""
    try:
        text = []
        pdf_reader = PdfReader(pdf_file, strict=False)
        for page in pdf_reader.pages:
            chunk = page.extract_text()
            if chunk:
                text.append(' '.join(chunk.split()))
        return ' '.join(text)
    except Exception as e:
        raise ValueError(f"Error processing PDF: {str(e)}")

def process_text(raw_text, user_question, api_key):
    """Process text with LLM"""
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=20000,
            chunk_overlap=1000,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.create_documents([raw_text])
        
        if not chunks:
            raise ValueError("No text content found to process")

        # Convert chunks to text
        chunk_texts = [doc.page_content for doc in chunks]
        
        # Get embeddings
        embeddings = get_embeddings_model(api_key)
        query_embedding = embeddings.embed_query(user_question)
        
        # Perform similarity search
        chunk_embeddings = [embeddings.embed_query(chunk) for chunk in chunk_texts]
        similarities = [
            np.dot(query_embedding, chunk_emb) / 
            (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb))
            for chunk_emb in chunk_embeddings
        ]
        
        # Get most relevant chunks
        top_indices = np.argsort(similarities)[-2:]
        relevant_chunks = [chunk_texts[i] for i in top_indices]
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
        
        # Create prompt
        template = """
        Context information is below.
        ---------------------
        {context}
        ---------------------
        Using only the context information provided above and not any prior knowledge, answer the following question:
        Question: {question}
        
        Answer:
        """
        
        # Process query
        context = "\n".join(relevant_chunks)
        prompt = template.format(context=context, question=user_question)
        messages = [HumanMessage(content=prompt)]
        
        # Get response
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        raise ValueError(f"Error processing text: {str(e)}")

def process_audio(audio, question, api_key):
    """Process audio file and answer question"""
    try:
        transcription = transcribe_audio(audio)
        if not transcription.strip():
            raise ValueError("No text was extracted from the audio file")
        return process_text(transcription, question, api_key)
    except Exception as e:
        raise ValueError(f"Error processing audio: {str(e)}")

def process_pdfs(pdfs, question, api_key):
    """Process PDF files and answer question"""
    try:
        extracted_texts = []
        for pdf in pdfs:
            text = get_pdf_text(pdf)
            if text.strip():  # Only add non-empty texts
                extracted_texts.append(text)
        
        if not extracted_texts:
            raise ValueError("No text could be extracted from the PDF files")
            
        combined_text = ' '.join(extracted_texts)
        return process_text(combined_text, question, api_key)
    except Exception as e:
        raise ValueError(f"Error processing PDFs: {str(e)}")

@app.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@app.route('/audio', methods=['GET', 'POST'])
def audio():
    """Audio processing route"""
    if request.method == 'POST':
        try:
            if 'audio' not in request.files:
                raise ValueError("No audio file uploaded")
            
            audio = request.files['audio']
            if not audio or not allowed_file(audio.filename, ALLOWED_EXTENSIONS_AUDIO):
                raise ValueError("Invalid audio file format. Only MP3 and WAV files are allowed.")
            
            question = request.form.get('question')
            if not question:
                raise ValueError("Please provide a question")
            
            google_key, _ = get_api_keys()
            if not validate_api_key(google_key):
                raise ValueError("Invalid or missing Google API key. Please update it in API Keys page.")
            
            response = process_audio(audio, question, google_key)
            return render_template('audio.html', response=response)
        except Exception as e:
            error_message = str(e)
            return render_template('audio.html', error=error_message), 400
            
    return render_template('audio.html')

@app.route('/pdf', methods=['GET', 'POST'])
def pdf():
    """PDF processing route"""
    if request.method == 'POST':
        try:
            if 'pdf' not in request.files:
                raise ValueError("No PDF files uploaded")
            
            files = request.files.getlist('pdf')
            if not files or len(files) == 0:
                raise ValueError("No PDF files selected")
            
            for file in files:
                if not file.filename or not allowed_file(file.filename, ALLOWED_EXTENSIONS_PDF):
                    raise ValueError(f"Invalid file format: {file.filename}. Only PDF files are allowed.")
            
            question = request.form.get('question')
            if not question:
                raise ValueError("Please provide a question")
            
            google_key, _ = get_api_keys()
            if not validate_api_key(google_key):
                raise ValueError("Invalid or missing Google API key. Please update it in API Keys page.")
            
            response = process_pdfs(files, question, google_key)
            return render_template('pdf.html', response=response)
        except Exception as e:
            error_message = str(e)
            return render_template('pdf.html', error=error_message), 400
            
    return render_template('pdf.html')

@app.route('/apikey', methods=['GET', 'POST'])
def apikey():
    """API key management route"""
    if request.method == 'POST':
        try:
            new_google_key = request.form.get('google_api_key')
            new_hf_key = request.form.get('hf_api_key')
            
            # Validate API keys
            if not all([new_google_key, new_hf_key]):
                raise ValueError("Both API keys are required")
            
            if not all(validate_api_key(key) for key in [new_google_key, new_hf_key]):
                raise ValueError("Invalid API key format")
            
            # Update global variables
            global google_api_key, hf_api_key
            google_api_key = new_google_key
            hf_api_key = new_hf_key
            
            flash("API keys updated successfully", 'success')
            return render_template('apikey.html', message="API keys updated successfully")
        except Exception as e:
            error_message = str(e)
            flash(error_message, 'error')
            return render_template('apikey.html', error=error_message), 400
            
    return render_template('apikey.html')

@app.route('/general', methods=['GET', 'POST'])
def general():
    """General question answering route"""
    if request.method == 'POST':
        try:
            question = request.form.get('question')
            if not question:
                raise ValueError("Please provide a question")
            
            _, hf_key = get_api_keys()
            if not validate_api_key(hf_key):
                raise ValueError("Invalid or missing Hugging Face API key. Please update it in API Keys page.")
            
            API_URL = "https://api-inference.huggingface.co/models/google/gemma-1.1-7b-it"
            headers = {"Authorization": f"Bearer {hf_key}"}
            response = requests.post(
                API_URL,
                headers=headers,
                json={
                    "inputs": f"Question: {question}\nAnswer:",
                    "parameters": {"max_length": 1024, "temperature": 0.3}
                }
            )
            
            if response.status_code != 200:
                raise ValueError("Error getting response from model")
                
            result = response.json()[0]['generated_text']
            return render_template('general.html', response=result)
        except Exception as e:
            error_message = str(e)
            return render_template('general.html', error=error_message), 400
            
    return render_template('general.html')

@app.route('/check-api-keys')
def check_api_keys():
    """Check API key status"""
    google_key, hf_key = get_api_keys()
    if validate_api_key(google_key) and validate_api_key(hf_key):
        return jsonify({
            'status': 'valid',
            'message': 'API keys are set'
        })
    return jsonify({
        'status': 'invalid',
        'message': 'One or more API keys are invalid or not set'
    }), 401

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500

@app.errorhandler(413)
def file_too_large(error):
    return render_template('error.html', error=f"File too large. Maximum size is {MAX_CONTENT_LENGTH/(1024*1024)}MB"), 413

if __name__ == '__main__':
    app.run(debug=False)