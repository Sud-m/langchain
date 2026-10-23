from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)  # Placeholder for future messages we get
from langchain_openai import ChatOpenAI

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a viral twitter influence grading a tweet. Generate critique and recommendations for the user's tweet
            Always provide detailed recommendations including requests for length, virality, style, etc."""
        ),
        MessagesPlaceholder(
            variable_name="messages"
        ),  # dynamically injects pre existing list of chat messages into a conversation prompt
        # When reflection prompt is used in invoke, we will set 'messages' to our chat messages
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a twitter Counter Strike influencer assistant tasked with writing excellent twitter posts.
            Generate the best twitter post possible for the user's request
            If the user provides critiques, respond with a revised version of your previous attempts.
            SIMPLY PROVIDE THE TWEET AND NO FURTHER COMMENTS"""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini")

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
