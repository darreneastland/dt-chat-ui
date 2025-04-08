import streamlit as st

st.set_page_config(page_title="Upload Documents", page_icon="📁", layout="centered")

st.title("📁 Upload Documents")
st.markdown("Upload PDFs, DOCX, or text files to embed them into the Digital Twin knowledge base.")

# File uploader
uploaded_files = st.file_uploader(
    label="Drag and drop or browse to upload files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        st.success(f"Uploaded {file.name}")
        # Placeholder: in future we'll add embedding logic here
