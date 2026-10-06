import httpx
import streamlit as st
from api_client import chat, ingest_document, list_documents, list_models

st.set_page_config(page_title="aie-local-llm", page_icon="🖥️")
st.title("aie-local-llm — local QA/RAG tester")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Engine")
    try:
        models = list_models()
    except httpx.HTTPError as exc:
        models = []
        st.error(f"Could not reach the API: {exc}")
    model = st.selectbox("Model", models) if models else None

    st.header("RAG")
    rag_enabled = st.toggle("Use retrieval-augmented generation", value=False)
    corpus = st.text_input("Corpus", value="default")
    top_k = st.slider("top_k", min_value=1, max_value=10, value=5)

    st.header("Documents")
    doc_text = st.text_area("Paste text to ingest")
    if st.button("Ingest") and doc_text.strip():
        try:
            result = ingest_document(doc_text, corpus=corpus)
            st.success(f"Ingested {result['chunks_created']} chunk(s) into '{corpus}'.")
        except httpx.HTTPError as exc:
            st.error(f"Ingest failed: {exc}")

    if st.button("List documents in corpus"):
        try:
            docs = list_documents(corpus)
            st.write(f"{len(docs)} chunk(s) in '{corpus}'")
            for d in docs:
                st.caption(d["text"][:200])
        except httpx.HTTPError as exc:
            st.error(f"Listing failed: {exc}")

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.caption(f"({source['score']:.2f}) {source['text']}")

question = st.chat_input("Ask a question")
if question:
    if model is None:
        st.error("No model available. Is the engine running and the API reachable?")
    else:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        messages = [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.history
        ]
        with st.chat_message("assistant"):
            try:
                response = chat(
                    model, messages, rag_enabled=rag_enabled, corpus=corpus, top_k=top_k
                )
                answer = response["choices"][0]["message"]["content"]
                sources = response.get("sources", [])
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.caption(f"({source['score']:.2f}) {source['text']}")
                st.session_state.history.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except httpx.HTTPError as exc:
                st.error(f"Request failed: {exc}")
