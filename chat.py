#!/usr/bin/env python3
"""Streamlit interface for the NASA mission intelligence RAG system."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

import llm_client
import rag_client
import ragas_evaluator

st.set_page_config(
    page_title="NASA Mission Intelligence",
    page_icon="🚀",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def initialize_rag_system(chroma_dir: str, collection_name: str, api_key: str):
    """Cache a collection connection for one backend/key combination."""
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["CHROMA_OPENAI_API_KEY"] = api_key
    return rag_client.initialize_rag_system(chroma_dir, collection_name)


def display_evaluation_metrics(scores: dict[str, Any]) -> None:
    """Render evaluation scores with explicit error handling."""
    if "error" in scores:
        st.warning(str(scores["error"]))
        return
    columns = st.columns(len(scores))
    for column, (metric_name, value) in zip(columns, scores.items(), strict=True):
        if isinstance(value, (int, float)):
            column.metric(metric_name.replace("_", " ").title(), f"{value:.3f}")


def display_sources(message: dict[str, Any]) -> None:
    """Show the retrieved evidence retained with an assistant turn."""
    sources = message.get("sources") or []
    if not sources:
        return
    with st.expander(f"Retrieved evidence ({len(sources)} excerpts)"):
        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata", {})
            source_name = metadata.get("file_path") or metadata.get("source", "Unknown")
            st.markdown(
                f"**Source {index}:** `{source_name}`  \n"
                f"Mission: `{metadata.get('mission', 'unknown')}` · "
                f"Category: `{metadata.get('document_category', 'unknown')}`"
            )
            document = source.get("document", "")
            st.caption(document[:700] + ("…" if len(document) > 700 else ""))


def main() -> None:
    st.title("🚀 NASA Mission Intelligence")
    st.caption(
        "Source-grounded answers from Apollo 11, Apollo 13, and Challenger mission archives"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Configuration")
        openai_key = st.text_input(
            "OpenAI API key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Used for query embeddings, answer generation, and optional evaluation.",
        )
        model = st.text_input(
            "Answer model",
            value="gpt-4o-mini",
            help="Any chat-completions model available to your OpenAI project.",
        )
        top_k = st.slider("Retrieved excerpts", min_value=1, max_value=10, value=4)
        mission_label = st.selectbox(
            "Mission scope",
            ["All", "Apollo 11", "Apollo 13", "Challenger"],
        )
        enable_evaluation = st.checkbox(
            "Run RAGAS after each answer",
            value=False,
            help="Adds evaluator LLM and embedding calls, increasing latency and API usage.",
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    backends = rag_client.discover_chroma_backends()
    if not backends:
        st.error("No ChromaDB collection was found.")
        st.code(
            "python embedding_pipeline.py --data-path ./data_text "
            "--chroma-dir ./chroma_db_openai",
            language="bash",
        )
        st.stop()

    backend_keys = list(backends)
    selected_key = st.sidebar.selectbox(
        "Document collection",
        backend_keys,
        format_func=lambda key: backends[key]["display_name"],
    )
    selected_backend = backends[selected_key]

    if not openai_key:
        st.info("Enter an OpenAI API key in the sidebar to begin.")
        st.stop()

    with st.spinner("Connecting to the mission archive…"):
        collection, success, error = initialize_rag_system(
            selected_backend["directory"],
            selected_backend["collection_name"],
            openai_key,
        )
    if not success:
        st.error(f"Unable to initialize retrieval: {error}")
        st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                display_sources(message)
                if message.get("evaluation"):
                    display_evaluation_metrics(message["evaluation"])

    prompt = st.chat_input("Ask about a NASA mission…")
    if not prompt:
        return

    user_turn = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_turn)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching NASA archives and composing an answer…"):
            try:
                result = rag_client.retrieve_documents(
                    collection,
                    prompt,
                    n_results=top_k,
                    mission_filter=mission_label,
                )
                documents = (result.get("documents") or [[]])[0]
                metadatas = (result.get("metadatas") or [[]])[0]
                distances = (result.get("distances") or [[]])[0]
                sources = rag_client.prepare_retrieved_sources(
                    documents,
                    metadatas,
                    distances,
                )
                context = rag_client.format_context(documents, metadatas, distances)
                response = llm_client.generate_response(
                    openai_key,
                    prompt,
                    context,
                    st.session_state.messages[:-1],
                    model,
                )
            except Exception as exc:
                st.error(f"The request failed: {exc}")
                st.session_state.messages.pop()
                return

        st.markdown(response)
        assistant_turn: dict[str, Any] = {
            "role": "assistant",
            "content": response,
            "sources": sources,
        }
        display_sources(assistant_turn)

        if enable_evaluation and documents:
            with st.spinner("Evaluating faithfulness and relevancy…"):
                scores = ragas_evaluator.evaluate_response_quality(
                    prompt,
                    response,
                    [context],
                    api_key=openai_key,
                )
            assistant_turn["evaluation"] = scores
            display_evaluation_metrics(scores)

    st.session_state.messages.append(assistant_turn)


if __name__ == "__main__":
    main()
