import streamlit as st
import os
from datetime import datetime
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.vectorstores import Pinecone  # avoid name clash
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# === PAGE CONFIG ===
st.set_page_config(page_title="Upload Reference Documents to DT", page_icon="📁")
st.title("📁 Upload Reference Documents to DT")
st.markdown("Drop documents here to add to the Digital Twin's memory.")

# === FILE UPLOAD ===
uploaded_files = st.file_uploader("Upload one or more documents", type=["pdf", "docx", "txt"], label_visibility="collapsed", accept_multiple_files=True)

if uploaded_files:
    openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENV") or st.secrets.get("PINECONE_ENV")
    index_name = "dt-knowledge"

    pc = PineconeClient(api_key=pinecone_api_key, environment=pinecone_env)

    # Optional index creation safeguard
    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.3, openai_api_key=openai_api_key)
    tagging_prompt = PromptTemplate.from_template("""
    Given the following text, classify its content into relevant high-level tags that could help categorize the document. Return a comma-separated list of tags only.
    
    TEXT:
    {chunk}
    """)
    tagger_chain = LLMChain(llm=llm, prompt=tagging_prompt)

    for uploaded_file in uploaded_files:
        st.write(f"Uploaded: {uploaded_file.name}")
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif uploaded_file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif uploaded_file.name.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                st.error(f"Unsupported file type: {uploaded_file.name}")
                continue

            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(docs)

            timestamp = datetime.utcnow().isoformat()
            inferred_type = "strategy" if "strategy" in uploaded_file.name.lower() else "general"

            for doc in split_docs:
                tag_response = tagger_chain.run(chunk=doc.page_content[:1000])
                doc.metadata.update({
                    "source_file": uploaded_file.name,
                    "uploaded_by": "Darren Eastland",
                    "uploaded_at": timestamp,
                    "document_type": inferred_type,
                    "tags": tag_response
                })

            Pinecone.from_documents(
                documents=split_docs,
                embedding=embeddings,
                index_name=index_name
            )

            st.success(f"✅ {uploaded_file.name} embedded and uploaded to DT memory with AI-generated tags.")

        except Exception as e:
            st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")

# === FOOTER ===
st.markdown("---")
st.caption("v1.25 – Multi-file Upload with AI Metadata Tagging – Digital Twin Chat Assistant – Darren Eastland")
