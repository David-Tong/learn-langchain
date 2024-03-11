# llm
import boto3
from langchain.llms.bedrock import Bedrock

# prompt
from langchain.prompts import PromptTemplate

# chain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# llm
client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

llm = Bedrock(
    model_id="amazon.titan-text-express-v1",
    client=client,
)

# prompt
prompt_template = """
System: The following is a friendly conversation between a knowledgeable helpful assistant and a customer.
The assistant is talkative and provides lots of specific details from it's context.

Current conversation:
{history}

User: {input}
Bot:
"""

prompt = PromptTemplate(
    input_variables=["history", "input"], template=prompt_template
)

# chain
memory = ConversationBufferMemory(human_prefix="User", ai_prefix="Bot")
conversation = ConversationChain(
    prompt=prompt,
    llm=llm,
    verbose=True,
    memory=memory,
)

# test
input = "How is Shanghai?"
print(conversation.predict(input=input))
