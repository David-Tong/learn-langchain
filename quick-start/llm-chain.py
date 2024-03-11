# llm
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()

# prompt
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

# output_parser
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

# chain
chain = prompt | llm | output_parser
print(chain.invoke({"input": "how can langsmith help with testing?"}))



