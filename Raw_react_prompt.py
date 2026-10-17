from dotenv import load_dotenv
import re
import inspect

load_dotenv()
from langsmith import traceable

import ollama  # From langchain-ollama

MAX_ITERATIONS = 3  # Your choice
MODEL = "qwen3.5:9b"

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


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


def get_tool_descriptions(tools_dict):
    descriptions = []
    for toolname, toolfunction in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers
        originalfunc = getattr(toolfunction, "__wrapped__", toolfunction)
        signature = inspect.signature(toolfunction)
        docstring = inspect.getdoc(toolfunction) or ""
        descriptions.append(f"{toolname}{signature} - {docstring}")
    return "\n".join(descriptions)


tooldescriptions = get_tool_descriptions(tools)
toolnames = ", ".join(tools.keys())


# Stays the same but no tools for llm, completely manual this time
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_trace(model, messages, options):
    return ollama.chat(model=MODEL, messages=messages, options=options)


react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tooldescriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{toolnames}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for i in range(MAX_ITERATIONS):
        print(f"Iteration {i+1}")
        fullprompt = prompt + scratchpad  # Create full prompt with past tool messages

        response = ollama_chat_trace(  # send to llm
            model=MODEL,
            messages=[{"role": "user", "content": fullprompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content
        print(f"LLM Output: {output}")

        print(" [Parsing] - Looking for final answer in LLM output")
        final_answer_match = re.search(
            r"Final Answer:\s*(.+)", output
        )  # See if llm is done
        # With all tool calls
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"    Final answer: {final_answer}")
            return final_answer

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        toolname = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        print(f"    [Tool selected] {toolname} with args: {tool_input_raw}")
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {toolname}({args})...")
        if toolname not in tools:
            observation = f"Error: Tool '{toolname}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[toolname](*args))

        print(f"  [Tool Result] {observation}")

        # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append).
        scratchpad += f"{output}\nObservation: {observation}\nThought:"
    print("Error: Max Iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Start langchain agent")
    result = run_agent(
        "What is the price for a laptop after applying the gold discount"
    )
