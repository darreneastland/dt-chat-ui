from datetime import datetime
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
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

    try:
        print("✅ Writing to memory:", reply_text[:200], "..." if len(reply_text) > 200 else "")
        vectorstore.add_documents([doc])
        print("✅ Memory write complete.")
    except Exception as e:
        print("❌ Memory write failed:", e)

