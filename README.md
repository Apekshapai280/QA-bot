# 🧠 Conversational QA Bot (Multi-Answer RAG Chatbot)

This project is an intelligent **Conversational QA Bot** designed to read technical documents in **PDF, DOCX, or TXT** format and provide accurate, multi-step answers to user queries.  
It leverages **Retrieval-Augmented Generation (RAG)** and **document chunk scoring** to retrieve, rank, and respond from contextually relevant information.

---

## ✅ Features

- 📄 Upload and parse documents (`.pdf`, `.docx`, `.txt`)  
- 🧩 Extract and chunk text with semantic embeddings  
- ⚡ Use **Gemma-2-9B-IT** via **Groq API** for fast, smart answering  
- 🧠 Contextual retrieval using LangChain’s history-aware retriever  
- 📊 Rank **multiple answers** based on cosine similarity  
- 💬 Store multi-session chat history  
- 🌐 Built with **Streamlit** for interactive frontend  

---

## 🛠️ Tech Stack

| Layer         | Technology                        |
|---------------|------------------------------------|
| **Frontend**  | Streamlit                          |
| **LLM**       | Gemma 2 9B via Groq API (`langchain_groq`) |
| **Vector DB** | ChromaDB with HuggingFace embeddings |
| **Embedding** | `all-MiniLM-L6-v2`                 |
| **Loaders**   | `PyPDFLoader`, `TextLoader`, `Docx2txtLoader` |
| **Chat Memory** | `ChatMessageHistory` from LangChain |
| **RAG**       | History-aware Retriever + QA Chain |

---

##  Getting Started

### 1. Clone the Repository

git clone https://github.com/Apekshapai280/QA-bot.git
cd QA-bot

### 2.Create a Virtual Environment
python -m venv venv
venv\Scripts\activate  # for Windows
# or
source venv/bin/activate  # for Mac/Linux

### 3.Install Dependencies
pip install -r requirements.txt

### 4. Setup Environment Variables
Create a .env file in the root directory:
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token

### 5.Run the App
streamlit run app.py

## Usage
Upload one or more .pdf, .docx, or .txt files.

Enter a Session ID to keep track of your conversation.

Ask questions based on the uploaded documents.

View top 4 ranked answers with scores based on cosine similarity.

Expand to view chat history and follow-up context.
