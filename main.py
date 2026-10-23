from dotenv import load_dotenv

load_dotenv()

from typing import (
    TypedDict,  # For structured dict with hints for keys, used when defining state schema
    Annotated,  # Helps add metadata to type hints
)
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import (
    add_messages,
)  # Ensure new messages are added to convo history (simply appends to list)
from chains import generation_chain, reflection_chain


class MessageGraph(TypedDict):
    messages: Annotated[
        list[BaseMessage], add_messages
    ]  # add_messages basically tells langgraph how to update the list
    # Basically messages is of type list[BaseMessage]


# Node names
REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph):
    return {
        "messages": [generation_chain.invoke({"messages": state["messages"]})]
    }  # Invoke generation with all messages we have so far
    # First iteration will have only user input
    # Following iterations will have critiques


def reflection_node(state: MessageGraph):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {
        "messages": [HumanMessage(content=res.content)]
    }  # Casting ai response to a human message to make the llm think the user is 'critiquing' the output


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(
    state: MessageGraph,
):  # NOT A NODE, it's a function of a conditional edge
    """Will be called everytime after we run generation node"""
    if (
        len(state["messages"]) > 3
    ):  # Simple logic, if number of messages is 6, we can end convo
        return END
    return REFLECT


builder.add_conditional_edges(
    GENERATE,
    should_continue,
    {  # This is a path map, these outputs come from should_continue and tell GENERATE where to go
        REFLECT: REFLECT,  # a:b, a is output from should_continue, b is corresponding node
        END: END,
    },
)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    print("Reflection Agent")
    input = HumanMessage(
        content="""
            Make this tweet better:
            @CounterStrike, your new reload features are garbage, slow animations suck, 
            provides no competitive advantage, and full mag reloads that reset mag count
            instead of adding bullets from your reserve supply is a pain to worry about. Please consider reverting both
            """
    )
    res = graph.invoke({"messages": [input]})
    print(res["messages"][-1].content)