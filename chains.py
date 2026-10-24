import datetime
from dotenv import load_dotenv
from langchain_openai.decisions import Answer

load_dotenv()
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,  # Parses tools from openai response
    PydanticToolsParser,
)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schemas import AnswerQuestion, ReviseAnswer

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher.
                Current time: {time}
                1. {first_instruction}
                2. Reflect and critique your answer. Be severe to maximize improvement
                3. Recommend search queries to research information and improve your answer.""",
        ),  # SystemMessage doesn't work with partial()
        MessagesPlaceholder(variable_name="messages"),  # history of messages
    ]
).partial(  # Helps populate the specified placeholders
    time=lambda: datetime.datetime.now().isoformat()
)

llm = ChatOpenAI(model="gpt-4o-mini")
parser = JsonOutputToolsParser(
    return_id=True
)  # Returns us the function call we got from llm and convert to dictionary
parser_pydantic = PydanticToolsParser(
    tools=[AnswerQuestion]
)  # takes llm response, searches for function calling invocation, converts it into an AnswerQuestion object

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)  # tool_choice forces llm to always use AnswerQuestion, hence grounding response in output format

revise_instructions = """Revise your previous answer using the new information
- You should use the previous critique to add important information to your answer.
- You must include numerical citations to ensure it can be verified.
- Add a "References" section to the bottom of your answer (does not count to word limit)
- [1] https://example.com
- [2] https://example.com
- You should use the previous critique to remove superfluous information from your answer and make sure it's cleanly around 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions,
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

if __name__ == "__main__":
    # Example. Main invocations of both chains happens in main.py
    human_message = HumanMessage(
        content="Write about the titanic, the main pitfall it encountered, what could have been done better, and future impact on ships."
    )
    chain = first_responder | parser_pydantic

    res = chain.invoke(input={"messages": [human_message]})
    print(res)
