from socket import if_indextoname
from dotenv import load_dotenv
import os

load_dotenv()

from langchain_unstructured import UnstructuredLoader  # No langchain community
from langchain_text_splitters import (
    CharacterTextSplitter,
)  # Takes text and splits it in to chunks
from langchain_openai import OpenAIEmbeddings  # Embedding model
from langchain_pinecone import PineconeVectorStore  # Persistent storage


if __name__ == "__main__":
    print("Ingesting data")
    loader = UnstructuredLoader(
        file_path="blog.txt", chunking_strategy="basic"
    )
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)  # List of documents
    print(f"created {len(texts)} chunks")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ.get("INDEX_NAME")
    )
