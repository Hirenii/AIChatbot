
# 🧠 Company AI Chatbot

An AI-powered chatbot designed for internal company use. It leverages Google's Gemini model via LangChain, FAISS-based retrieval, and real-time messaging with Socket.IO and AioHTTP.

## 🚀 Features

* Real-time chat via WebSockets
* Google Gemini (1.5 Flash) LLM integration
* Semantic search using FAISS + HuggingFace embeddings
* Contextual responses based on vector-retrieved documents
* CORS-enabled AioHTTP backend
* Modular, production-ready code structure

## 📁 Project Structure

```
.
├── app.py                     # AioHTTP + Socket.IO server
├── chatbot.py                 # LangChain logic and Gemini integration
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not committed)
├── faiss_index/               # Local FAISS vector store
```

## 🔧 Tech Stack

* Python 3.8+
* [AioHTTP](https://docs.aiohttp.org/) + [python-socketio](https://python-socketio.readthedocs.io/)
* [LangChain](https://docs.langchain.com/) with `langchain-google-genai`
* [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
* [HuggingFace Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/company-ai-chatbot.git
cd company-ai-chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your `.env` file

Create a `.env` file in the root directory and add your API key:

```
GOOGLE_GENAI_API_KEY=your_google_api_key
```

### 5. Ensure FAISS index is built

Make sure `faiss_index/` exists at the specified path and contains indexed documents.

### 6. Run the server

```bash
python app.py
```

The server will run at `http://0.0.0.0:8765/` and support WebSocket connections.

## 📡 Socket.IO Events

| Event Name         | Direction       | Payload                    | Description                    |
| ------------------ | --------------- | -------------------------- | ------------------------------ |
| `connect`          | server → client | "Welcome!"                 | Connection confirmation        |
| `message`          | client → server | user message (str)         | Handles user input             |
| `message`          | server → client | `{ response: "...", ... }` | Bot response                   |
| `get_chat_history` | client → server | none                       | Request chat history           |
| `chat_history`     | server → client | list of chat entries       | Full chat log for this session |

## 🧪 Example Client (Python)

```python
import socketio

sio = socketio.Client()
sio.connect("http://localhost:8765")

@sio.on("message")
def on_message(data):
    print("Bot:", data)

sio.emit("message", "Does this company serve AI solutions?")
```

