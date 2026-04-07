from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "generate a linkedin post on the topic:{topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "generate a X  post on the topic:{topic}",
    input_variables=['topic']
)

chain = RunnableParallel({
    'tweet':RunnableSequence(prompt1,model,parser),
    'linkedin':RunnableSequence(prompt2,model,parser)
})

result=chain.invoke({'topic':'cinematography'})

print(result)