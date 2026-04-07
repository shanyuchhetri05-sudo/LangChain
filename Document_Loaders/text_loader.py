from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")  # safer current model
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="write a summary for the following poem: {poem}",
    input_variables=["poem"]
)

# keep loader separate
loader = TextLoader("demo.txt", encoding="utf-8")

# convert loader → text
load_text = RunnableLambda(lambda _: loader.load()[0].page_content)

# chain
chain = load_text | prompt1 | model | parser

# invoke (no input needed → we ignore input with lambda)
result = chain.invoke({})

print(result)
