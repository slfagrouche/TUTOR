# TUTOR
![alt text](![image](h[ttps://github.com/slfagrouche/TUTOR/assets/105510022/11a0fedf-a723-4cd5-8115-f31c4631f3f3](https://github-production-user-asset-6210df.s3.amazonaws.com/105510022/341986987-11a0fedf-a723-4cd5-8115-f31c4631f3f3.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240622%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240622T173331Z&X-Amz-Expires=300&X-Amz-Signature=ea658c98265c3e28069c6e215d24232b921e74d101fefdd2bed7150570a3a36b&X-Amz-SignedHeaders=host&actor_id=105510022&key_id=0&repo_id=806923151))

)

TUTOR is an AI-powered educational assistant designed to help users with audio transcription, PDF text extraction, and question answering. It leverages advanced AI models and Retrieval-Augmented Generation (RAG) to provide accurate and contextually relevant responses based on uploaded content.

## Features

- **Audio Transcription**: Upload an audio file for transcription.
- **PDF Text Extraction**: Extract text from uploaded PDF files.
- **Question Answering**: Get detailed responses to questions based on the content of audio or PDF files using RAG.
- **General AI Tutor**: Ask general questions and receive AI-generated responses.

## Pitch: Why TUTOR is Different

Finding accurate and contextually relevant educational information can be challenging with traditional LLM applications. TUTOR leverages Retrieval-Augmented Generation (RAG) to combine the power of large language models with a curated knowledge base. This ensures that the responses are not only generated based on the AI's training data but also grounded in the specific content provided by the user, whether it be from audio transcriptions or PDF text extractions. This approach significantly enhances the accuracy and relevance of the information provided.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/slfagrouche/TUTOR.git
   ```
2. Navigate to the project directory:
   ```sh
   cd TUTOR
   ```
3. Install the dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory and add your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key
   HF_API_KEY=your_huggingface_api_key
   ```

## Usage

1. Run the Flask app:
   ```sh
   python app.py
   ```
2. Open your browser and go to `http://127.0.0.1:5000/`.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact Our Team

- Said Lfagrouche - [Linkedin](https://www.linkedin.com/in/saidlfagrouche/)
- Saad Ahmad Sabri - [Linkedin](https://www.linkedin.com/in/saad-ahmad-sabri-a42669208/)
