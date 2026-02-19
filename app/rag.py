# rag.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama

# ===============================
# GLOBAL STATE
# ===============================

chat_history = []
current_uploaded_file = None

VECTOR_DB_PATH = "faiss_index"

# Use temperature=0 for strict factual answers
embedding = OllamaEmbeddings(model="tinyllama")
llm = ChatOllama(
    model="tinyllama",
    temperature=0  # 🔥 reduces hallucination
)

# ===============================
# DOCUMENT PROCESSING
# ===============================

def create_vector_store(pdf_path):
    global current_uploaded_file

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,     # smaller chunk = cleaner retrieval
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(docs, embedding)
    vector_store.save_local(VECTOR_DB_PATH)

    current_uploaded_file = os.path.basename(pdf_path)
    chat_history.clear()


def load_vector_store():
    if not os.path.exists(VECTOR_DB_PATH):
        raise Exception("No document uploaded yet.")

    return FAISS.load_local(
        VECTOR_DB_PATH,
        embedding,
        allow_dangerous_deserialization=True
    )


def get_uploaded_filename():
    return current_uploaded_file


# ===============================
# STRICT PROMPT TEMPLATE
# ===============================

def build_strict_prompt(context, question):
    return f"""
Use ONLY the information provided below to answer the question.

If the answer is not clearly stated in the text,
respond exactly with:
I don't have that information in the uploaded documents.

TEXT:
{context}

Question: {question}

Answer:
"""


# ===============================
# ONE-OFF QUESTION
# ===============================

def ask_question(query):
    try:
        vector_store = load_vector_store()

        # reduce retrieval noise
        docs = vector_store.similarity_search(query, k=2)

        if not docs:
            return "I don't have that information in the uploaded documents."

        context = "\n".join([doc.page_content for doc in docs])

        # 🔍 DEBUG (optional)
        # print("Retrieved context:\n", context)

        prompt = build_strict_prompt(context, query)

        response = llm.invoke(prompt)

        answer = response.content.strip()

        # Extra safety check (optional strict guard)
        if len(answer) > 800:
            return "I don't have that information in the uploaded documents."

        return answer

    except Exception:
        return "Please upload a document first."


# ===============================
# CHAT WITH MEMORY
# ===============================

def chat_with_context(user_input):
    try:
        vector_store = load_vector_store()

        docs = vector_store.similarity_search(user_input, k=2)

        if not docs:
            return "I don't have that information in the uploaded documents."

        context = "\n".join([doc.page_content for doc in docs])

        # Same simple prompt as ask_question (avoids TinyLlama echoing long "CRITICAL RULES")
        prompt = build_strict_prompt(context, user_input)

        response = llm.invoke(prompt)
        answer = response.content.strip()

        chat_history.append(f"User: {user_input}")
        chat_history.append(f"Assistant: {answer}")

        return answer

    except Exception:
        return "Please upload a document first."
