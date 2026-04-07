from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')

template = load_prompt("template.json")

prompt= template.invoke({
    'paper_input':'Attention all you need',
    'style_input':'begginer friendly',
    'length_input':'1 or 2 pragraph'
    })


result = model.invoke(prompt)

print(result.text)