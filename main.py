import streamlit as st
from typing import List, Dict, Any
from backend.core import run_llm


def _format_sources(context_docs: List[Any]) -> List[str]:
    """Simply obtain sources if any from list of documents"""
    return list(
        dict.fromkeys(
            str(meta.get("source") or "Unknown")
            for doc in context_docs
            if (meta := getattr(doc, "metadata", None)) is not None
        )
    )


st.set_page_config(page_title="Documentation Helper", layout="centered")

st.title("Langchain Documentation Helper")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain Docs. I'll retrieve the relevant context and cite sources",
            "sources": ["www.langchain.com"],
        }
    ]
# Get each message
for msg in st.session_state.messages:
    with st.chat_message(
        msg["role"]
    ):  # msg["role"] changes icon, everything in this will be one chat message
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):  # Renders as drop down
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about LangChain")
if prompt:  # If user pressed enter
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer"):
                result: Dict[str, Any] = run_llm(prompt)
                answer = result["answer"] or "No answer returned"
                sources = _format_sources(result["context"])

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

        except Exception as e:
            st.error("Failed to generate a response")
            st.exception(e)
