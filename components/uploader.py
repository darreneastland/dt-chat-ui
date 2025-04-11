import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import settings
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

def load_and_split(file_path, file_type):
    if file_type == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == ".docx":
        loader = Docx2txtLoader(file_path)
    elif file_type == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(documents)

def store_embeddings(docs, namespace="default"):
    vectorstore = Pinecone.from_existing_index(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY),
        namespace=namespace
    )
    vectorstore.add_documents(docs)
    return vectorstore

def summarise_doc_excerpt(docs, filename):
    text_excerpt = "\n".join([d.page_content for d in docs[:2]])[:1000]
    return f"Filename: {filename}\nExcerpt:\n{text_excerpt}"

