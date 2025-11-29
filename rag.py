

import os

import streamlit as st
from dotenv import load_dotenv

import chromadb

from google import genai


# === 1. Load API key ===
load_dotenv()

# Try environment variable first (local or cloud)
api_key_google = os.getenv("GOOGLE_API_KEY")

# Fallback: try Streamlit secrets (Cloud)
if not api_key_google:
    api_key_google = st.secrets.get("GOOGLE_API_KEY")

# If still missing, stop app with a nice error message
if not api_key_google:
    st.error("GOOGLE_API_KEY not found. Set it in a .env file (locally) or in Streamlit Secrets (on the cloud).")
    st.stop()

client = genai.Client(api_key=api_key_google)

# === 2. Chroma client & collection (load existing index) ===
chroma_client = chromadb.PersistentClient(path="mkdocs_db/")

google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=api_key_google,
    model_name="models/text-embedding-004",
)

collection = chroma_client.get_or_create_collection(
    name="MkDocs",
    embedding_function=google_ef,
)

# === 3. RAG logic (same as in notebook) ===

SYSTEM_PROMPT = """
You are an AI assistant specialized ONLY in answering questions about MkDocs,
the static site generator for project documentation.

Rules:
- Use ONLY the information given in the retrieved context from the MkDocs docs.
- If the answer is not clearly supported by the context, say:
  "I don't know based on the MkDocs documentation I have."
- Do NOT answer questions about anything outside MkDocs (no general chat, no other frameworks).
- If the user asks about something unrelated to MkDocs, politely refuse and say that you only support MkDocs documentation questions.
- Prefer short, clear, step-by-step explanations when appropriate.
"""

def build_context_block(result):
    docs = result["documents"][0]
    metas = result["metadatas"][0]

    context_str = ""
    for i, (text, meta) in enumerate(zip(docs, metas), start=1):
        path = meta.get("path", "")
        context_str += f"Source {i} (path: {path}):\n{text}\n\n"
    return context_str

def answer_question(query: str, k: int = 4):
    # 1) retrieve from Chroma
    result = collection.query(
        query_texts=[query],
        n_results=k,
    )

    context_str = build_context_block(result)

    # 2) human prompt
    human_prompt = f"""
You are given some context taken from the official MkDocs documentation:

{context_str}

User question: {query}

Instructions:
- Answer ONLY using the context above.
- If the context is not enough, say you don't know.
- If the question is not about MkDocs, refuse and say you only answer MkDocs questions.
- When possible, mention which source paths you used.
"""

    # 3) call Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[SYSTEM_PROMPT, human_prompt],
    )

    answer = response.text
    source_paths = [m.get("path", "") for m in result["metadatas"][0]]

    # also return the raw chunks for UI if you like
    docs = result["documents"][0]

    return answer, list(zip(source_paths, docs))


# === 4. Streamlit UI ===

st.set_page_config(page_title="MkDocs RAG Assistant", page_icon="📚")

st.title("📚 MkDocs RAG Assistant")
st.write("Ask questions about MkDocs documentation. The assistant only answers using the official MkDocs docs embedded in ChromaDB.")

query = st.text_area("Your question about MkDocs:", height=80, placeholder="Example: How do I deploy MkDocs to GitHub Pages?")

col1, col2 = st.columns(2)
with col1:
    k = st.slider("Number of neighbors (k)", min_value=1, max_value=8, value=4, step=1)
with col2:
    run_button = st.button("Ask", type="primary")

if run_button and query.strip():
    with st.spinner("Thinking..."):
        try:
            answer, sources = answer_question(query.strip(), k=k)
        except Exception as e:
            st.error(f"Error while answering: {e}")
        else:
            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources from MkDocs docs")
            for i, (path, text) in enumerate(sources, start=1):
                with st.expander(f"Source {i}: {path}"):
                    st.write(text)
