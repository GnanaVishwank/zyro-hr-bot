"""
Zyro Dynamics HR Help Desk — Streamlit Chatbot
Deploy on https://share.streamlit.io

Required secrets (Settings -> Secrets on Streamlit Cloud):
    GROQ_API_KEY = "..."
    LANGCHAIN_API_KEY = "..."   # optional but enables tracing

Folder structure expected:
    app.py
    requirements.txt
    data/  <- the 11 HR policy PDFs (copy them in from the Kaggle dataset)
"""

import os
import glob
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_DIR = "data"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
TOP_K = 4
SIMILARITY_THRESHOLD = 0.30
REFUSAL_MESSAGE = "I can only answer HR-related questions from Zyro Dynamics policy documents."

SYSTEM_PROMPT = """You are the Zyro Dynamics HR Help Desk assistant.
Answer ONLY using the provided context from Zyro Dynamics HR policy documents.
Rules:
- If the answer is in the context, answer clearly and concisely.
- If the context does NOT contain the answer, say exactly:
  "I can only answer HR-related questions from Zyro Dynamics policy documents."
- Never invent policy details that aren't in the context.
- Keep answers factual and grounded in the retrieved text.

Context:
{context}"""

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🧑‍💼", layout="centered")

# --------------------------------------------------------------------------
# Secrets / env vars
# --------------------------------------------------------------------------
os.environ.setdefault("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "")))
if st.secrets.get("LANGCHAIN_API_KEY", os.environ.get("LANGCHAIN_API_KEY")):
    os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", os.environ.get("LANGCHAIN_API_KEY"))
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

if not os.environ.get("GROQ_API_KEY"):
    st.error("Missing GROQ_API_KEY. Add it under Settings → Secrets on Streamlit Cloud.")
    st.stop()


# --------------------------------------------------------------------------
# Build RAG pipeline once, cache across reruns
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing HR policy documents...")
def build_pipeline():
    pdf_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdf_paths:
        st.error(f"No PDFs found in ./{DATA_DIR}/. Add the 11 HR policy PDFs to that folder.")
        st.stop()

    raw_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        doc_name = os.path.basename(path)
        for p in pages:
            p.metadata["source"] = doc_name
        raw_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(
        chunks, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 20, "lambda_mult": 0.5},
    )

    llm = ChatGroq(model=LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(f"[{d.metadata.get('source', '?')}] {d.page_content}" for d in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return vectorstore, retriever, rag_chain


vectorstore, retriever, rag_chain = build_pipeline()


def is_in_scope(question: str) -> bool:
    results = vectorstore.similarity_search_with_score(question, k=1)
    if not results:
        return False
    _, score = results[0]
    return score >= SIMILARITY_THRESHOLD


def ask_hr_bot(question: str) -> dict:
    if not is_in_scope(question):
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)

    if REFUSAL_MESSAGE.split(".")[0] in answer:
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    sources = sorted({d.metadata.get("source", "unknown") for d in docs})
    return {"answer": answer, "sources": sources}


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🧑‍💼 Zyro Dynamics HR Help Desk")
st.caption("Ask me about leave, WFH, code of conduct, benefits, POSH, onboarding, travel & expense, and more — answered straight from Zyro's HR policy documents.")

with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This assistant only answers questions covered by Zyro Dynamics' 11 HR "
        "policy documents. Anything else gets a polite refusal."
    )
    st.divider()
    st.subheader("Try asking:")
    st.write("- How many earned leave days do I get per year?")
    st.write("- What's the hybrid work-from-home policy?")
    st.write("- How do I file a POSH complaint?")
    st.write("- What's the notice period during probation?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s}")

user_input = st.chat_input("Ask an HR question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Checking HR policy documents..."):
            result = ask_hr_bot(user_input)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("📄 Sources"):
                for s in result["sources"]:
                    st.write(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
