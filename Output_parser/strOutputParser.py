from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

template1= PromptTemplate(
    template="generate a detail note on the {topic}",
    input_variables=['topic']
)

template2= PromptTemplate(
    template="generate a 5 line summary of the text: {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result=chain.invoke({'topic':'covid19'})

print(result)

