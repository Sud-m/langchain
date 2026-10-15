from dotenv import load_dotenv

load_dotenv()
from langsmith import traceable

import ollama  # From langchain-ollama

MAX_ITERATIONS = 3  # Your choice
MODEL = "qwen3.5:0.8b"

# THIS EXAMPLE IS MAINLY FOR OLLAMA, VARIES PER VENDOR
# WHOLE POINT IS THAT THIS EXAMPLE ISN'T VIABLE TO REPLICATE EVERYTIME, HENCE LANGCHAIN


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog.
    Accepted product names: laptop, keyboard, Mouse, Headphones. There are no models or products apart from these 4
    """
    print(f"    >> Executing get_product_price({product})")
    prices = {"laptop": 1599, "keyboard": 100, "mouse": 45, "headphones": 40}
    return prices.get(product, -1)


@traceable(run_type="tool")
def apply_discount(price: int, discount_tier: str) -> float:  # Type hints are important
    """Applies a discount tier to a price and returns the final price
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount({price}, {discount_tier})")
    discounts = {"bronze": 5, "silver": 10, "gold": 15}
    discount = discounts.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]
# Ollama can auto generate these schemas if you pass the functions directly as tools (like @tool in langchain)
#   tools_for_llm = [get_product_price, apply_discount]
# But you have to follow google docstring format
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# Essentially this whole bit of jargon is automated by @tool


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_trace(messages):
    return ollama.chat(
        model=MODEL, tools=tools_for_llm, messages=messages, options={"temperature": 0}
    )


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    # tools_dict = {t.name: t for t in tools}  # Cannot use without langchain
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }
    print(f"Question: {question}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. You have access to a product catalog tool and a discount tool\n"
                "STRICT RULES - You must follow these exactly: \n"
                "1. NEVER guess or assume product price\n"
                "2. Only call apply_discount AFTER you have recieved a price from get_product_price\n"
                "Pass the exact price returned by get_product_price\n"
                "3. Never apply discounts yourself, always use apply_discount\n"
                "If the user doesn't specify a discount tier, ask them which one to use, do not assume\n"
            ),
        },
        {
            "role": "user",
            "content": question,
        },  # Ollama uses user for user input role, others diff, langchain does this for us
    ]
    for i in range(MAX_ITERATIONS):
        print(f"Iteration {i+1}")

        # use ollama.chat() directly instead of llmwithtools.invoke()
        response = ollama_chat_trace(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        # No tool calls? Final answer
        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call - force one tool call per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"[Tool selected]: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use(
            **tool_args
        )  # ** for unpacking the dictionary into keyword arguments
        print(f"[Tool result]: {observation}")

        messages.append(ai_message)
        messages.append({"role": "tool", "content": str(observation)})

    print("Error: Max Iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Start langchain agent")
    result = run_agent(
        "What is the price for a laptop after applying the gold discount"
    )
