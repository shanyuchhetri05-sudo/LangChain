from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.messages import AIMessage,HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

prompt_template = ChatPromptTemplate([
    ('system','you are a {domain} expert'),
    ('human','explain in simple terms what is {topic}')
])

prompt = prompt_template.invoke({'domain':'cricket','topic':'doosra'})

result = model.invoke(prompt)
prompt_template.append(('ai',result.text))

print(result.text)

print(prompt_template)