import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain_core import vectorstores

load_dotenv()
from langchain.agents import create_agent
from langchain.chat_models import (
    init_chat_model,
)  # Receives string and returns chat model

from langchain.messages import ToolMessage  # Retrieval tool
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)
model = init_chat_model("gpt-5", model_provider="openai")


@tool(response_format="content_and_artifact")  # will return two return values
def retrieve_context(query):
    """Retrieve relevant documentation to help answer user queries about Langchain"""
    # We don't use similarity_search() from pinecone as it's not as clean with langsmith
    # retrieved_docs = vector_store.similarity_search(query, k=4)
    retrieved_docs = vector_store.as_retriever().invoke(query, k=4)  # top 4 chunks

    # Serialize the documents for the model
    # Essentially a big string to attach to prompt
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get("source", "Unknown")}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs  # content, artifact


def run_llm(query):
    """Runs the RAG retrieval pipeline to answer a query using retrieved documentation\n

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
            answer: The generated answer
            content: list of retrieved documents"""

    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation."
        "You have access to a tool that retrieves relevant documentation"
        "Use the tool to find relevant information before answering questions."
        "Always cite the sources you use in your answers."
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    # Build messages list
    messages = [{"role": "user", "content": query}]
    response = agent.invoke({"messages": messages})
    answer = response["messages"][-1].content  # last message and its content

    # Get context documents form Tool artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if the tool message has an artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of document objects
            if isinstance(message.artifact, list):
                context_docs.append(message.artifact)

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    result = run_llm(query="What are deep agents?")
    print(result["answer"])
