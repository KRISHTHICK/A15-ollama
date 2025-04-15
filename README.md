To create a **simple RAG-based chatbot using Ollama** (running locally without API keys) that:

- Lets the user **upload a document** (PDF or TXT),
- Extracts and stores the **content locally**,
- Performs **retrieval-augmented generation (RAG)** using **Ollama LLM**,
- Runs entirely via **Streamlit in VS Code**,

Here's a **complete working code** example for you:

---

### ✅ Prerequisites

Make sure you have:

```bash
pip install streamlit langchain langchain-community chromadb pypdf ollama
```

Also, download a model in Ollama (like `gemma`, `llama2`, `mistral`, etc.):

```bash
ollama pull gemma
```

---

### 📁 `app.py` (Streamlit Chatbot)

```python
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os
import tempfile

# Title
st.title("📄 RAG Chatbot using Ollama (Local & No API Key)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a document (PDF)", type=["pdf"])

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file:
    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name

    # Load & Split PDF
    loader = PyPDFLoader(tmp_pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Embedding + Vector Store
    embeddings = OllamaEmbeddings(model="gemma")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)

    # LLM and RAG Chain
    llm = Ollama(model="gemma")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    # User Input
    user_question = st.text_input("Ask a question about your document:")

    if user_question:
        result = qa_chain({"query": user_question})
        answer = result["result"]

        # Show response
        st.markdown("### 🤖 Answer")
        st.write(answer)

        # Save chat history
        st.session_state.history.append(("You", user_question))
        st.session_state.history.append(("Bot", answer))

    # Display chat history
    if st.session_state.history:
        st.markdown("### 💬 Chat History")
        for speaker, message in st.session_state.history:
            st.write(f"**{speaker}:** {message}")
```

---

### ▶️ To Run in VS Code Terminal

```bash
streamlit run app.py
```

---

### 💡 Notes

- No API key is needed.
- Uses `Ollama` locally for embedding and answering.
- You can upload **any PDF** file, and it will analyze it with local RAG.

---

Let me know if you want:
- TXT, DOCX file support
- Multi-PDF support
- Local storage of chat history

Or help packaging it as an executable!
