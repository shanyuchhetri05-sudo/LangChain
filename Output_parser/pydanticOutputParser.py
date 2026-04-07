from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Annotated

class Person(BaseModel):
    name: str
    age: int = Field(gt=0,lt=120)
    city: str

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give give info of name,age and city of a fictional person /n {output_format}",
    partial_variables={'output_format':parser.get_format_instructions()}
)


chain = template | model | parser

print("result:",chain.invoke({}))