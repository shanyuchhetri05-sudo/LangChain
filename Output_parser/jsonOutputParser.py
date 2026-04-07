from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

parser = JsonOutputParser()

template = PromptTemplate(
    template="give me the name , age and city of an fictional person \n {output_format_instruction}",
    input_variables=[],
    partial_variables={'output_format_instruction':parser.get_format_instructions()}
)

prompt = template.format()

print("promt:",prompt)

result = model.invoke(prompt)

print("result:",result.content)