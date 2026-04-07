from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableLambda,RunnablePassthrough,RunnableParallel,RunnableBranch
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Generate a detailed report the topic:{topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "summarize the follwing report:{report}",
    input_variables=['report']
)

report_generation_chain =RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x:len(x.split())>300,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_generation_chain,branch_chain)

result = final_chain.invoke({'topic':'US vs IRAN'})

print(result)