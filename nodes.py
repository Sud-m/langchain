from dotenv import load_dotenv

load_dotenv()
from langgraph.graph import (
    MessagesState,
)  # MessagesState is a state that stores the messages in the conversation
from langgraph.prebuilt import ToolNode  # Node for executing tools
from langchain_core.messages import SystemMessage
from react import llm, tools

SYSTEM_MESSAGE = """
                 You are a helpful assistant that can use tools to answer questions.
"""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """Run the agent reasoning node."""
    response = llm.invoke(
        input=[SystemMessage(content=SYSTEM_MESSAGE), *state["messages"]]
    )  # First iteration will have only human message, second will have llm response as well, etc
    return {"messages": [response]}


tool_node = ToolNode(tools)
