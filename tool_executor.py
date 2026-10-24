from dotenv import load_dotenv

load_dotenv()

from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import (
    ToolNode,
)  # A node we can invoke that will look in state for last message and see if there are any tool calls, if there are it will execute them
from schemas import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=5)

# Create two tools, answer question tool and revise answer tool, both will run TavilySearch, hence import from schemas
# This is good for langsmith tracing too, helps differentiate between answer and revise

def run_queries(search_queries: list[str], **kwargs):
    """Run the genereated queries"""
    return tavily_tool.batch([{"query": query} for query in search_queries])



execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name="AnswerQuestion"),
        StructuredTool.from_function(run_queries, name="ReviseAnswer"),
    ]    
)