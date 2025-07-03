import streamlit as st
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv
from huggingface_hub import login
import tempfile

# Load environment variables
load_dotenv()
login(os.getenv("HF_TOKEN"))

# Setup embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

# Streamlit UI
st.set_page_config(page_title="QA Chatbot", layout="centered")
st.title("📚👨‍💻Conversational QA Bot")
st.write("Upload PDF, DOCX, or TXT files. Ask questions, get smart multi-step answers.")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ GROQ_API_KEY not set in .env file")
    st.stop()

llm = ChatGroq(api_key=api_key, model="Gemma2-9b-It")

session_id = st.text_input("Session ID", value="default_session")
if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Upload files", type=["pdf", "txt", "docx"], accept_multiple_files=True)
documents = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        if ext == "pdf":
            loader = PyPDFLoader(tmp_path)
        elif ext == "txt":
            loader = TextLoader(tmp_path)
        elif ext == "docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            st.warning(f"Unsupported file type: {ext}")
            continue

        documents.extend(loader.load())

    # Create vectorstore
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Create contextualization and QA prompts
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and latest question, return a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    answer_prompt = PromptTemplate.from_template("""
    You are an assistant helping with questions based on a technical document.
    Answer the following question based **only** on the provided context.

    Question: {question}
    Context: {context}

    Answer:
    """)

    answer_chain = answer_prompt | llm | StrOutputParser()

    # ✅ FIXED FUNCTION: Generate multiple answers with relevance scoring
    def generate_multi_answers(question, top_docs):
        answers = []
        question_embedding = embeddings.embed_query(question)

        for doc in top_docs:
            context = doc.page_content
            response = answer_chain.invoke({"question": question, "context": context})
            context_embedding = embeddings.embed_documents([context])[0]
            score = cosine_similarity([question_embedding], [context_embedding])[0][0]
            answers.append({
                "answer": response.strip(),
                "score": round(score, 4),
                "source": doc.metadata.get("source", "N/A")
            })

        sorted_answers = sorted(answers, key=lambda x: x['score'], reverse=True)
        return sorted_answers

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Ask a question:")
    if user_input:
        session_history = get_session_history(session_id)
        st.chat_message("user").write(user_input)

        # Manual top-K doc retrieval for multi-answer
        top_docs = retriever.get_relevant_documents(user_input, k=4)
        ranked_answers = generate_multi_answers(user_input, top_docs)

        with st.chat_message("assistant"):
         for idx, item in enumerate(ranked_answers):
          st.markdown(f"""
        <div style="border: 1px solid #d3d3d3; border-radius: 10px; padding: 10px; margin-bottom: 10px;">
            <strong>Answer {idx+1}:</strong><br>
            {item['answer']}<br>
            <span style="font-size: 0.8em; color: gray;">Source: {item['source']}</span>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.store[session_id].add_user_message(user_input)
        st.session_state.store[session_id].add_ai_message(ranked_answers[0]["answer"])

        with st.expander("💬 Chat History"):
            for msg in session_history.messages:
                st.markdown(f"**{msg.type.title()}**: {msg.content}")
else:
    st.info("⬆️ Upload one or more documents to begin")
