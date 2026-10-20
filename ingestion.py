from importlib import metadata
from dotenv import load_dotenv
import os
import ssl
from typing import Any, Dict, List
from rich.console import console
import openai

load_dotenv()

import asyncio
import certifi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import (
    TavilyCrawl,
    TavilyExtract,
    TavilyMap,
)

# from langchain_chroma import (
#     Chroma,
# )  # Chroma vector store, not needed if we have pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import (
    Document,
)  # Represents a text document with metadata

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

# Configure ssl context to use certifi certificates
# Mainly to make a bunch of requests with Tavily and not get ssl certificate error
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=True,
    chunk_size=50,  # Chunk size for the embeddings
    retry_min_seconds=10,  # Wait 10 seconds after a retry
)  # embeddings class
# chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings) For local chroma

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()  # Takes a url and extracts a doc


def chunk_urls(urls, chunk_size=3):
    """Split URLs into chunks of specified size"""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunks.append(urls[i : i + chunk_size])
    return chunks


async def extract_batch(urls, batch_num):
    """Extract documents from a batch of urls"""
    try:
        console.print(
            f"TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs",
            style="bold blue",
        )
        docs = await tavily_extract.ainvoke(
            input={"urls": urls}
        )  # TavilyExtract() instance
        res = docs.get("results", [])
        console.print(
            f"Batch {batch_num} completed, extracted {len(res)} documents",
            style="bold green",
        )
        return res
    except Exception as e:
        console.print(f"Failed to extract batch {batch_num}: {e}")
        return []


async def async_extract(url_batches):
    console.print("DOCUMENT EXTRACTION PHASE", style="bold yellow")
    console.print(
        f"Starting concurrent extraction of {len(url_batches)} batches",
        style="bold white",
    )
    tasks = [extract_batch(batch, i+1) for i, batch in enumerate(url_batches)]

async def main():
    """Main async function that orchestrates the whole process"""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    # log_info(
    #     "  Tavily Crawl: Starting to crawl documentation from python.langchain.com"
    # )

    # res = tavily_crawl.invoke(
    #     {
    #         "url": "https://docs.langchain.com/oss/python/langchain",
    #         "max_depth": 2,  # How far down the website to crawl
    #         "extract_depth": "advanced",  # Extracts advanced information from page
    #         "instructions": "content on AI agents",  # Filters based on this instruction
    #     }
    # )
    # all_docs = [
    #     Document(page_content=result["raw_content"], metadata={"source": result["url"]})
    #     for result in res["results"]
    # ]
    # log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site")

    # Website discovery (TavilyMap)
    log_info(
        "  TavilyMap: Starting to map documentation structure from https://docs.langchain.com/oss/python/langchain",
        Colors.PURPLE,
    )
    sitemap = tavily_map.invoke("https://docs.langchain.com/oss/python/langchain")
    log_success(
        f"  TavilyMap: Successfully mapped {len(sitemap["results"])} URLs from documentation site"
    )

    # URL Batching
    url_batches = chunk_urls(sitemap["results"], 10)
    log_info(
        f"URL Processing: Split {len(sitemap["results"])} URLs into {len(url_batches)} batches",
        Colors.BLUE,
    )


if __name__ == "__main__":
    asyncio.run(main())
