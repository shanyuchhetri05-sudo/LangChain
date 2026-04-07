from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "write a joke about topic:{topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "explain the context of the joke:{joke}",
    input_variables=['joke']
)

joke_generator_chain = RunnableSequence(prompt1,model,parser)


parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(joke_generator_chain,parallel_chain)

result = final_chain.invoke({'topic':'fart'})

print(result)