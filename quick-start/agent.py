# web based documents loader
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()

# openai embedding
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# build documents into vectors
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# retriever
retriever = vector.as_retriever()

# retriever tool
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

# search
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults()

# tool chain
tools = [retriever_tool, search]

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
print(agent_executor.invoke({"input": "how can langsmith help with testing?"}))
print(agent_executor.invoke({"input": "what is the weather in SF?"}))

from langchain_core.messages import HumanMessage, AIMessage
chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]

print(agent_executor.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
}))
