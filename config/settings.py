# config/settings.py

import os

class Settings:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENV = os.getenv("PINECONE_ENV")
        self.PINECONE_INDEX_NAME = "dt-knowledge"

# Instantiate and expose
settings = Settings()
