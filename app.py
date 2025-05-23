import streamlit as st
import tempfile
import sys
import os
sys.path.append(os.path.dirname(__file__))

from config import GROQ_API_KEY, GROQ_MODEL
from utils.pdf_loader import load_and_split_pdf
from utils.vector_store import create_vectorstore
from utils.qa_chain import get_conversational_chain

st.set_page_config(page_title="📚 AI Research Assistant", layout="wide")
st.title("📄 AI Research Assistant")
st.markdown("Upload a research paper (PDF) and ask questions!")

if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    with st.spinner("Processing PDF..."):
        docs = load_and_split_pdf(file_path)
        vectorstore = create_vectorstore(docs)
        st.session_state.chain = get_conversational_chain(vectorstore)
        st.success("✅ PDF processed! Ask your question below.")

if st.session_state.chain:
    query = st.text_input("Ask something about the PDF")
    if query:
        with st.spinner("Thinking..."):
            response = st.session_state.chain.run({"question": query})
            st.session_state.chat_history.append((query, response))

    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 AI:** {a}")
