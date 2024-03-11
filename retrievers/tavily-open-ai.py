# search
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults()

# tool chain
tools = [search]

# prompt, llm, agent
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# invoke
print(agent_executor.invoke({"input": "You are a professional consultant, please use tools to search and verify the question, Is Trump current US president?"}))

"""
# invoke
response = agent_executor.invoke({"input": "What is the weather in Austin?"})
print("input : {}".format(response["input"]))
print("output : {}".format(response["output"]))

from langchain_core.messages import HumanMessage, AIMessage
chat_history = [HumanMessage(content="What is the weather in Austin?"), AIMessage(content="It is extremly hot in summer!")]

print(agent_executor.invoke({
    "chat_history": chat_history,
    "input": "Tell me which month is best for travel."
}))
"""