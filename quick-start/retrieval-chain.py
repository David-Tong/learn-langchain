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

# llm
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()

# prompt
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# retriever
from langchain.chains import create_retrieval_chain
retriever = vector.as_retriever()

# build chain
from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# output
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])