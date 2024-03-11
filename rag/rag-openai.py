# search
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults()

# llm
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

# prompt
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

# chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {"context": search, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("What is Task Decomposition?"))
