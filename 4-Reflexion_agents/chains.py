from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime 
from euriai.langchain import EuriaiChatModel
from schema import AnswerQuestion
from langchain_core.messages import HumanMessage

import os 
from dotenv import load_dotenv
load_dotenv()

llm = EuriaiChatModel(
    api_key=os.getenv("EURI_API_KEY"),
    model="gpt-4.1-nano"
)

## Actor Agent Prompt
actor_prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system",
         """ You are an expert AI researcher.

    Current time: {time}

    1. {first_instruction}
    2. Reflect and critique your answer.Be severe to maximize improvement.
    3. After the reflection, **list 1-3 search queries seperately** for researching improvements.Do not include them inside the reflection.
         """,
         ),
         MessagesPlaceholder(variable_name="messages"),
         ("system","Answer the user's question above using the required format.")
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder_prompt_template=actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion]) 

response= first_responder_chain.invoke({
    "messages":[HumanMessage(content="AI agents taking over the content creation")]
})

print(response)
