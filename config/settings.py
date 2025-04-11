import os

class settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your-openai-key"
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "your-pinecone-key"
    PINECONE_ENV = os.getenv("PINECONE_ENV") or "your-env"
    PINECONE_INDEX_NAME = "dt-knowledge"

