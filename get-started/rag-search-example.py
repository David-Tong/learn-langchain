# rag search
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma

# retriever
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
texts = ["Harrison worked at Kensho", "Bears like to eat honey"]
retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever()

# chain
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | model | output_parser

print(chain.invoke("where did harrison work?"))
