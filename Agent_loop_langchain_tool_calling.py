from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages.tool import tool_call
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.chat_models import (
    init_chat_model,
)  # utility function in langchain that initializes a langchain chat model with just a string

from langchain.tools import tool  # decorator
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    SystemMessage,
    HumanMessage,
)  # Messages for tools, llm, and user input respectively

MAX_ITERATIONS = 3  # Your choice

MODEL = "gpt-5"


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog.
    Accepted product names: laptop, keyboard, Mouse, Headphones. There are no models or products apart from these 4
    """
    print(f"    >> Executing get_product_price({product})")
    prices = {"laptop": 1599, "keyboard": 100, "Mouse": 45, "Headphones": 40}
    return prices.get(product, -1)


@tool
def apply_discount(price: int, discount_tier: str) -> float:  # Type hints are important
    """Applies a discount tier to a price and returns the final price
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount({price}, {discount_tier})")
    discounts = {"bronze": 5, "silver": 10, "gold": 15}
    discount = discounts.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


from langsmith import traceable


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}  # t.name comes from tool decorator
    llm = init_chat_model(f"openai:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(
        tools
    )  # Only works with llms that can call functions, works with all models if so regardless of vendor
    print(f"Question: {question}")

    messages = [
        SystemMessage(
            content="You are a helpful shopping assistant. You have access to a product catalog tool and a discount tool"
            "STRICT RULES - You must follow these exactly: "
            "1. NEVER guess or assume product price"
            "2. Only call apply_discount AFTER you have recieved a price from get_product_price"
            "Pass the exact price returned by get_product_price"
            "3. Never apply discounts yourself, always use apply_discount"
            "If the user doesn't specify a discount tier, ask them which one to use, do not assume"
        ),
        HumanMessage(content=question),
    ]
    for i in range(MAX_ITERATIONS):
        print(f"Iteration {i+1}")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:                               #This means the llm response contains no tool calls
                                                         # i.e. the llm knew it had to stop 
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content
        
        tool_call = tool_calls[0]    # not ideal, just an easy way to do it for now
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        
        print(f"[Tool Selected]: {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)
        print(f"    [Tool Result]: {observation}")
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id) # id is important to match specific result to one tool call
        )

    print("Error: Max Iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Start langchain agent")
    result = run_agent(
        "What is the price for a laptop after applying the gold discount"
    )
    print(result)
