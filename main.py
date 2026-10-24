from dotenv import load_dotenv

load_dotenv()
from typing import Literal
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import START, END, StateGraph, MessagesState
from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_ITERATIONS = 2
# Create chains for AI powered research functionality: Corresponds to Actor Agent (First responder chain)
# Enhance chains for answer revision capabilities: Revisor agent
# Add tools for graph nodes: ToolNode and executing tools
# Implement the complete message graph


def draft_node(state: MessagesState):
    """Draft the initial response"""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def revise_node(state: MessagesState):
    """Revise the answer based on tool results"""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine whether to continue or end based on iteration count"""
    tool_count = sum(isinstance(item, ToolMessage) for item in state["messages"])
    if tool_count > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)
builder.add_edge(
    START, "draft"
)  # First responder edge, takes user input and executes first responder agent
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges(
    "revise", event_loop, ["execute_tools", END]
)  # See if we should continue revising or not. If yes execute tools again, if not then end
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


if __name__ == "__main__":
    print("Reflexion Agent")
    res = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write about the titanic. list its pitfalls, things we learned, outcomes for future ships.",
                }
            ]
        }
    )

    last = res["messages"][-1]

    if isinstance(last, AIMessage) and last.tool_calls:
        print(last.tool_calls[0]["args"]["answer"])
