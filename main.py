import os
from dotenv import load_dotenv

load_dotenv()
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# for lecl
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

print("Initializing System")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-5")

vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)  # Retrieve only top 3 documents

prompt_template = ChatPromptTemplate.from_template(
    """
                Answer the question based only on the following context:
                {context}
                Question: {question}
                Provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


# To simply show the flow of things, naive implementation
def retrieval_change_without_lcel(query: str):
    """Simple retrieval chain without LCEL.\n
    Manually retrieve documents, formats them, generates response.
    - Manual step by step execution
    - No built in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error prone"""
    docs = retriever.invoke(
        query
    )  # Get most relevant documents, will show up in langsmith

    # Format docs into a string
    context = format_docs(docs)
    print("\n\nCONTEXT: " + context + "\n\n")
    # Format prompt with the context and question
    messages = prompt_template.format_messages(context=context, question=query)
    # Invoke the llm with the full message
    response = llm.invoke(messages)
    return response.content


def create_retrieval_chain_with_lcel():
    """Create a retrieval chain using LCEL.\n
    Returns a chain that can be invoked with {"question": "..."}
    """
    # format_docs will work, langchain converts regular python functions into runnable lambdas
    # format_docs -> RunnableLambda(format_docs)
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )  # Creates new dict, combined original input with computed field
    # {"question": "...", "context": chain output (formatted docs)}
    # question comes from invoke
    # Result of this will go into prompt_template under context key
    # Same as Imp 1, just chained

    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving")
    query = "What is Pinecone in machine learning?"

    # Raw invocation without RAG
    # print("Implementation 0 with raw llm invocation no RAG")
    # print("-" * 60)
    # result = llm.invoke([HumanMessage(content=query)])
    # print("LLM ANSWER: " + result.content)

    # # RAG Implementation without LCEL
    # print("Implementation 1: RAG Without LCEL")
    # print("-" * 60)
    # res = retrieval_change_without_lcel(query)
    # print("LLM ANSWER: " + res)

    # RAG Implementation with LCEL
    print("Implementation 2: RAG With LCEL")
    print("-" * 60)
    retrieval_chain = create_retrieval_chain_with_lcel()
    result = retrieval_chain.invoke({"question": query})
    print("LLM ANSWER: " + result)