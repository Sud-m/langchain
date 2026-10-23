from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import HumanMessagePromptTemplate
from langgraph.graph import MessagesState, StateGraph, END  # Initializes graph
from nodes import run_agent_reasoning, tool_node

AGENT_REASONING = "agent_reason"
ACT = "act"
LAST = -1  # Index of last message between user and agent


def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASONING, run_agent_reasoning)
flow.set_entry_point(AGENT_REASONING)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(
    AGENT_REASONING, should_continue, {END: END, ACT: ACT}
)  # Whatever comes from the should continue function, map it to go to the end node or to go to ACT
flow.add_edge(ACT, AGENT_REASONING)
app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Langgraph")
    city = input(
        "Enter the city whose temperature you want to know and tripled: "
    ).strip()
    question_template = HumanMessagePromptTemplate.from_template(
        "What is the temperature in {city}? List it and triple it."
    )
    question = question_template.format(city=city)
    res = app.invoke({"messages": [question]})
    print(res["messages"][LAST].content)
