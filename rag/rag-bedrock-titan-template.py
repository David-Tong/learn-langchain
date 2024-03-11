# search
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults()

# llm
import boto3
from langchain.llms.bedrock import Bedrock

client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
model_id = "amazon.titan-text-lite-v1"
llm = Bedrock(client=client, model_id=model_id, streaming=False)

# prompt
from langchain_core.prompts import ChatPromptTemplate
template = """
Answer the question based only on the following context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

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
