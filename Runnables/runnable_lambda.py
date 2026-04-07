from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableLambda,RunnablePassthrough,RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "write a joke about topic:{topic}",
    input_variables=['topic']
)

joke_generator_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'total words':RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(joke_generator_chain,parallel_chain)

result = final_chain.invoke({'topic':'cricket'}) 

print('JOKE : ',result['joke'])

print('WORD COUNT : ', result['total words'])

