from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import create_react_agent  # NEW import
from langchain_core.messages import HumanMessage


load_dotenv()
import os

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Tool
search_tool = TavilySearchResults()
tools = [search_tool]

# Create the agent (LangGraph)
agent = create_react_agent(llm, tools)

# Run it
result = agent.invoke({"messages":
                       [HumanMessage(content="What is the AQI of Delhi?")] 
                       })
print(result)
