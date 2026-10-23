# Reasoning engine
from dotenv import load_dotenv

load_dotenv()
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


@tool
def triple(num: float) -> float:
    """Triples a float number input"""
    return float(num) * 3


import requests


@tool
def get_weather(city_name: str) -> dict:
    """Get current weather for a city. Temperatures are in degrees Fahrenheit."""
    baseurl = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_name,
        "appid": os.environ["WEATHER_API_KEY"],
        "units": "imperial",
    }
    res = requests.get(baseurl, params=params, timeout=15)
    if not res.ok:
        # Avoid including the request URL, which contains the API key, in errors.
        raise RuntimeError(f"Weather API request failed (HTTP {res.status_code})")
    return res.json()


tools = [
    # TavilySearch(max_results=1),
    triple,
    get_weather,
]  # TavilySearch description exists by default

llm = ChatOpenAI(model="gpt-5", temperature=-0).bind_tools(tools=tools)
