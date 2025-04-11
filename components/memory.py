from datetime import datetime
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
# - from langchain_community.schema import Document
from config.settings import settings 


def get_vectorstore(index_name, namespace):
    return Pinecone.from_existing_index(
        index_name=index_name,
        embedding=OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY),
        namespace=namespace
    )

def store_to_memory(vectorstore, reply_text):
    doc = Document(
        page_content=reply_text,
        metadata={
            "type": "chat_summary",
            "source": "user_interaction",
            "timestamp": datetime.now().isoformat()
        }
    )
    vectorstore.add_documents([doc])

