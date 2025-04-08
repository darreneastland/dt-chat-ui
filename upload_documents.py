import streamlit as st
import pinecone
import openai
import os
from docx import Document
import PyPDF2
import tempfile
from typing import List

# === CONFIG ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
INDEX_NAME = "dt-knowledge"
ENVIRONMENT = "us-east-1"

pinecone.init(api_key=PINECONE_API_KEY, environment=ENVIRONMENT)
index = pinecone.Index(INDEX_NAME)
openai.api_key = OPENAI_API_KEY

# === FUNCTIONS ===
def load_docx(file) -> str:
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def load_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def embed_texts(text_chunks: List[str]) -> List[List[float]]:
    response = openai.Embedding.create(
        input=text_chunks,
        model="text-embedding-ada-002"
    )
    return [data["embedding"] for data in response["data"]]

def upload_to_pinecone(text_chunks: List[str], embeddings: List[List[float]], metadata_src: str):
    vectors = [
        {
            "id": f"{metadata_src}_{i}",
            "values": emb,
            "metadata": {"text": chunk, "source": metadata_src}
        }
        for i, (chunk, emb) in enumerate(zip(text_chunks, embeddings))
    ]
    index.upsert(vectors)

# === STREAMLIT UI ===
st.title("📚 Upload Knowledge to DT")
st.markdown("Upload the Digital Twin Charter or other files to inject them into DT memory.")

# Auto-load charter from local path
if "charter_uploaded" not in st.session_state:
    try:
        with open("Digital_Twin_Charter_v2_Revised.docx", "rb") as f:
            charter_text = load_docx(f)
        st.success("Digital Twin Charter loaded.")
        chunks = chunk_text(charter_text)
        embeddings = embed_texts(chunks)
        upload_to_pinecone(chunks, embeddings, "charter")
        st.session_state.charter_uploaded = True
    except Exception as e:
        st.warning(f"Charter not uploaded automatically: {e}")

# Drag & Drop
uploaded_files = st.file_uploader("Drop files here", type=["pdf", "docx", "txt"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        file_type = file.name.split(".")[-1]
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            if file_type == "pdf":
                content = load_pdf(tmp_path)
            elif file_type == "docx":
                content = load_docx(tmp_path)
            elif file_type == "txt":
                with open(tmp_path, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                st.error(f"Unsupported file type: {file_type}")
                continue

            chunks = chunk_text(content)
            embeddings = embed_texts(chunks)
            upload_to_pinecone(chunks, embeddings, file.name)
            st.success(f"Uploaded and embedded {file.name}")

        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
