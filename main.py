# from langchain import
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


class Source(BaseModel):
    url: str = Field(description="The url of the Source")


class AgentResponse(BaseModel):
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Search for 3 job postings for an AI engineer using langchain in the bay area on linkedin"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
