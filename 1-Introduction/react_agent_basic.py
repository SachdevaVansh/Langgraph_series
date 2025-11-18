from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
import datetime

# from langgraph.prebuilt import create_react_agent  # NEW import(Provided by LangGraph)

from langchain_core.messages import HumanMessage


load_dotenv()
import os

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Tool
search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

tools = [search_tool,get_system_time]

## Creating the agent using Langchain...
agent= initialize_agent(tools=tools,llm=llm,agent="zero-shot-react-description",verbose=True)

# Create the agent (LangGraph)
# agent = create_react_agent(llm, tools)

# Run it
result = agent.invoke({"input":
                       [HumanMessage(content="What is the AQI of Delhi?")] 
                       })
print(result)
