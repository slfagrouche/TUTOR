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

load_dotenv()

app = Flask(__name__)

# Load API keys from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
hf_api_key = os.getenv("HF_API_KEY")

# Initialize and configure the Whisper and PDF processing tools
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Whisper model
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=True)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Define the ASR pipeline
asr_pipeline = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                        feature_extractor=processor.feature_extractor, device=device)

def convert_mp3_to_wav(mp3_audio):
    """
    Convert an MP3 audio file to WAV format.

    Args:
        mp3_audio (FileStorage): MP3 audio file uploaded by the user.

    Returns:
        BytesIO: WAV audio file.
    """
    mp3_path = mp3_audio.stream
    mp3_sound = AudioSegment.from_file(mp3_path, format="mp3")
    buffer = io.BytesIO()
    mp3_sound.export(buffer, format="wav")
    buffer.seek(0)
    return buffer

def transcribe_audio(audio):
    """
    Transcribe an audio file to text using the ASR pipeline.

    Args:
        audio (FileStorage): Audio file uploaded by the user.

    Returns:
        str: Transcribed text from the audio.
    """
    if audio.filename.endswith('.mp3'):
        audio = convert_mp3_to_wav(audio)
    
    audio_data, sr = librosa.load(audio, sr=16000)
    result = asr_pipeline({"array": audio_data, "sampling_rate": sr})
    return result['text']

def get_pdf_text(pdf_file):
    """
    Extract text from a PDF file.

    Args:
        pdf_file (FileStorage): PDF file uploaded by the user.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def process_audio(audio, question):
    """
    Process an audio file and answer a question based on its content.

    Args:
        audio (FileStorage): Audio file uploaded by the user.
        question (str): User's question.

    Returns:
        str: Transcription and response based on the audio content.
    """
    transcription = transcribe_audio(audio)
    response = answer_question(question, None, transcription)
    return "Audio transcription and response: " + response

def process_pdfs(pdfs, question):
    """
    Process PDF files and answer a question based on their content.

    Args:
        pdfs (list of FileStorage): List of PDF files uploaded by the user.
        question (str): User's question.

    Returns:
        str: Extracted text and response based on the PDF content.
    """
    extracted_text = ""
    for pdf in pdfs:
        extracted_text += get_pdf_text(pdf)
    response = answer_question(question, extracted_text, None)
    return "PDF content and response: " + response

def answer_question(user_question, pdf_text, audio_text):
    """
    Answer a user's question based on provided PDF or audio content.

    Args:
        user_question (str): User's question.
        pdf_text (str): Text extracted from PDF files.
        audio_text (str): Transcribed text from audio files.

    Returns:
        str: Response to the user's question.
    """
    raw_text = (pdf_text if pdf_text else "") + (audio_text if audio_text else "")
    if raw_text == "":
        return "No content to process. Please upload a PDF or audio file."

    text_chunks = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000).split_text(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
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
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5, google_api_key=google_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, prompt=prompt)

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

@app.route('/')
def index():
    """
    Render the index page.

    Returns:
        str: Rendered HTML template for the index page.
    """
    return render_template('index.html')

@app.route('/audio', methods=['GET', 'POST'])
def audio():
    """
    Handle audio file uploads and process them to answer a user's question.

    Returns:
        str: Rendered HTML template with the response.
    """
    if request.method == 'POST':
        audio = request.files['audio']
        question = request.form.get('question')
        response = process_audio(audio, question)
        return render_template('audio.html', response=response)
    return render_template('audio.html', response=None)

@app.route('/pdf', methods=['GET', 'POST'])
def pdf():
    """
    Handle PDF file uploads and process them to answer a user's question.

    Returns:
        str: Rendered HTML template with the response.
    """
    if request.method == 'POST':
        pdfs = request.files.getlist('pdf')
        question = request.form.get('question')
        response = process_pdfs(pdfs, question)
        return render_template('pdf.html', response=response)
    return render_template('pdf.html', response=None)

@app.route('/general', methods=['GET', 'POST'])
def general():
    """
    Handle general questions and provide responses using an AI model.

    Returns:
        str: Rendered HTML template with the response.
    """
    if request.method == 'POST':
        question = request.form.get('question')
        response = answer_general_question(question)
        return render_template('general.html', response=response)
    return render_template('general.html', response=None)

def answer_general_question(user_question):
    """
    Answer a general question using an AI model.

    Args:
        user_question (str): User's question.

    Returns:
        str: AI-generated response to the question.
    """
    API_URL = "https://api-inference.huggingface.co/models/google/gemma-1.1-7b-it"
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    prompt = f"You are General AI Tutor, helping students with homework and general questions. Student question: {user_question}. Response: should be clear and detailed."

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({"inputs": prompt})
    return f"AI Response: '{output[0]['generated_text']}'"

if __name__ == '__main__':
    app.run(debug=True)


