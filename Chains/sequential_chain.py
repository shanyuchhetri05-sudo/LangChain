from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

template1 = PromptTemplate(
    template='generate a detailed report on the topic:{topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

template2 = PromptTemplate(
    template='generate a 5 points summary of the report:{report}',
    input_variables=['report']
)

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'supernova'})

chain.get_graph().draw_ascii()



