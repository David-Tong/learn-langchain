from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
import boto3
from langchain.llms.bedrock import Bedrock

search = TavilySearchResults()

template = """turn the following user input into a search query for a search engine:

{input}"""
prompt = ChatPromptTemplate.from_template(template)

client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
model_id = "amazon.titan-text-lite-v1"
model = Bedrock(client=client, model_id=model_id, streaming=False)

chain = prompt | model | search

print(chain.invoke({"input": "I'd like to figure out what games are tonight"}))
