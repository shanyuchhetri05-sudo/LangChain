from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel
from typing import Literal

class Feedback(BaseModel):

    sentiment: Literal['positive','negative']

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

parser = StrOutputParser()

parser1 = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template='analyze the sentiment of the feedback:{feedback} \n {output_format}',
    input_variables=['feedback'],
    partial_variables={'output_format':parser1.get_format_instructions}
)

prompt1 = PromptTemplate(
    template='generate a response on the negative feedback:{feedback} ',
    input_variables=['feedback']
) 

prompt2 = PromptTemplate(
    template='generate a response on the positive feedback:{feedback} ',
    input_variables=['feedback']
) 

sentiment_classifier_chain = prompt | model | parser1


response_chain = RunnableBranch(
    (lambda x:x.sentiment == 'negative',prompt1|model|parser),
    (lambda x:x.sentiment == 'positive',prompt2|model|parser),
    RunnableLambda(lambda x: 'could not find the sentiment')
)

chain = sentiment_classifier_chain | response_chain

result = chain.invoke({'feedback':'this is the dumbest phone'})

print(result)

print(chain.get_graph().draw_ascii())


