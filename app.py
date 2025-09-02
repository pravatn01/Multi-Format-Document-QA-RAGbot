import streamlit as st
import fitz
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from firestore_utils import save_message, load_messages

# ===================== CONFIG =====================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DIR = "chroma_db"

# ===================== PDF TEXT EXTRACTION (PyMuPDF) =====================
def get_pdf_text_with_metadata(pdf_docs):
    docs = []
    for pdf in pdf_docs:
        pdf_file = fitz.open(stream=pdf.read(), filetype="pdf")
        for page_num, page in enumerate(pdf_file):
            page_text = page.get_text("text") or ""
            if page_text.strip():
                docs.append(Document(
                    page_content=page_text,
                    metadata={"source": pdf.name, "page": page_num + 1}
                ))
        pdf_file.close()
    return docs

# ===================== CHUNKING =====================
def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    return text_splitter.split_documents(documents)

# ===================== VECTOR STORE =====================
def create_or_update_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    if not os.path.exists(CHROMA_DIR):
        vector_store = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    else:
        vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        vector_store.add_documents(chunks)
    vector_store.persist()
    return vector_store

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# ===================== CREATE CONVERSATIONAL CHAIN =====================
def get_conversation_chain(vector_store):
    prompt_template = """
Answer the question using the provided context. Be detailed and thorough.
If the answer is not in the context, say: "Answer is not available in the context."

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",
        temperature=0.3
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return conversation_chain

# ===================== STREAMLIT APP =====================
def main():
    st.set_page_config("Chat with PDF", layout="wide")
    st.title("📚 ChatGPT-style PDF QnA with Persistent Chat History (Firestore)")

    # Assign session_id per user
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Load vector store if available
    if "conversation" not in st.session_state:
        if os.path.exists(CHROMA_DIR):
            vector_store = load_vector_store()
            st.session_state.conversation = get_conversation_chain(vector_store)
        else:
            st.session_state.conversation = None

    # Load chat history from Firestore only once
    if "messages" not in st.session_state:
        st.session_state.messages = load_messages(st.session_state.session_id)

    # Sidebar - Upload PDFs
    with st.sidebar:
        st.header("📂 Upload & Process PDFs")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    docs = get_pdf_text_with_metadata(pdf_docs)
                    chunks = chunk_documents(docs)
                    vector_store = create_or_update_vector_store(chunks)
                    st.session_state.conversation = get_conversation_chain(vector_store)
                st.success("✅ PDFs processed and added to vector store.")
            else:
                st.warning("Please upload at least one PDF.")

    # Display stored chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources"):
                    for src in msg["sources"]:
                        st.write(src)

    # User input
    if prompt := st.chat_input("Ask something about your PDFs..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(st.session_state.session_id, "user", prompt)

        if st.session_state.conversation is None:
            st.warning("⚠ Please upload and process PDFs first.")
        else:
            with st.spinner("Thinking..."):
                result = st.session_state.conversation({"question": prompt})
                answer = result["answer"]
                sources = [
                    f"File: {doc.metadata['source']} | Page: {doc.metadata['page']}"
                    for doc in result.get("source_documents", [])
                ]

            # Show bot answer
            with st.chat_message("assistant"):
                st.markdown(answer)
                if sources:
                    with st.expander("📄 Sources"):
                        for src in sources:
                            st.write(src)

            # Save bot message to state & Firestore
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
            save_message(st.session_state.session_id, "assistant", answer, sources)

if __name__ == "__main__":
    main()
