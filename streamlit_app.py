import streamlit as st
from rag_pipeline import setup_pipeline, answer_question

st.set_page_config(page_title="Notes Q&A Bot", page_icon="📚", layout="centered")

st.title("📚 Notes Q&A Bot")
st.caption("RAG-powered study assistant — ask questions from your notes, get answers with sources")

if "vectordb" not in st.session_state:
    with st.spinner("Indexing your notes... (only takes a moment)"):
        st.session_state.vectordb = setup_pipeline()
    st.success("Bot is ready!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 View Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}]** {src['content_preview']}")
                    st.caption(f"From: {src['source_file']}")

if question := st.chat_input("Write your question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question(st.session_state.vectordb, question)
            st.markdown(result["answer"])
            with st.expander("📄 View Sources"):
                for i, src in enumerate(result["sources"], 1):
                    st.markdown(f"**[{i}]** {src['content_preview']}")
                    st.caption(f"From: {src['source_file']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })

with st.sidebar:
    st.header("⚙️ How it works")
    st.markdown("""
    1. **Load** — reads files from your notes folder
    2. **Chunk** — splits large text into smaller pieces
    3. **Embed** — converts each chunk into a vector
    4. **Store** — saves vectors in ChromaDB
    5. **Retrieve** — finds chunks similar to your question
    6. **Generate** — LLM creates answer using retrieved context
    """)

    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()