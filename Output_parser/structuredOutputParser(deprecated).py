from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

schema =    [
    ResponseSchema(name = 'fact_1',description = 'Fact1 about the topic'),
    ResponseSchema(name = 'fact_2',description = 'Fact2 about the topic'),
    ResponseSchema(name = 'fact_3',description = 'Fact3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema);

template = PromptTemplate(
    template=""""give any 3 facts about the topic {topic} /n {output_format}""",
    input_variables=['topic'],
    partial_variables={'output_format':parser.get_format_instructions()}
)

prompt = template.invoke({'topic':'supernova'})

result = model.invoke(prompt)

final_result = parser.parse(result)

print("prompt:",prompt)

print("final_result:",final_result.text)