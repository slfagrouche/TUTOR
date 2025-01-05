# **TUTOR Setup Instructions**

Welcome to **TUTOR**, your AI-powered educational assistant! This guide will walk you through setting up the project locally on your machine.

---

## 🛠️ **System Requirements**

- **Python Version:** 3.9.20
- **Operating System:** Windows, macOS, or Linux
- **Dependencies:** Listed in `requirements.txt`
- **Environment Variables:** API keys for integration

---

## 🐍 **1. Install Python 3.9.20**

1. Download Python 3.9.20 from the [official Python website](https://www.python.org/downloads/release).
2. Verify installation:
   ```sh
   python --version
   ```

---

## 💻 **2. Set Up Virtual Environment**

1. **Navigate to the Project Directory:**
   ```sh
   cd path/to/TUTOR
   ```
2. **Create a Virtual Environment:**
   ```sh
   python -m venv venv
   ```
3. **Activate the Virtual Environment:**
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```
   - On Windows:
     ```cmd
     .\venv\Scripts\activate
     ```
4. Verify the virtual environment is active:
   ```sh
   which python   # macOS/Linux
   where python   # Windows
   ```

---

## 📦 **3. Install Dependencies**

Ensure `pip` is up-to-date and install the required libraries:
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 **4. Configure Environment Variables**

1. Create a `.env` file in the root directory.
2. Add your API keys:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   HF_API_KEY=your_huggingface_api_key
   SECRET_KEY=your_choosing_secret_key_for_flask
   ```

---

## 🚀 **5. Run the Application**

1. Start the application:
   ```sh
   python app.py
   ```
2. Open your browser and go to:
   **http://127.0.0.1:5000/**

---

## 📝 **Troubleshooting**

- **Virtual Environment Not Activating:** Verify you're in the project directory.
- **Missing Dependencies:** Run `pip install -r requirements.txt` again.
- **API Errors:** Ensure your `.env` file is correctly set up.

---

For further assistance, reach out via [GitHub Issues](https://github.com/slfagrouche/TUTOR/issues).

Happy Coding! 🚀

