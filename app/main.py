# app/main.py
from fastapi import FastAPI, UploadFile, File, Body
import shutil, os
from app.rag import create_vector_store, ask_question, chat_with_context
from fastapi.middleware.cors import CORSMiddleware
from app.rag import get_uploaded_filename



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    # Overwrite if file exists
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    create_vector_store(file_path)
    return {"message": "PDF uploaded and processed successfully"}


@app.get("/ask")
def ask(query: str):
    answer = ask_question(query)
    return {"answer": answer}


@app.post("/chat")
def chat(user_input: str = Body(..., embed=True)):
    answer = chat_with_context(user_input)
    return {"answer": answer}


@app.get("/")
def home():
    return {"message": "Company Knowledge Chatbot running"}

@app.get("/current-file")
def current_file():
    filename = get_uploaded_filename()
    if not filename:
        return {"file": None}
    return {"file": filename}

