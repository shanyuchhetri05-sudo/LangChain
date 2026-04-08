from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

# load_dotenv()

# model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")  # safer current model
# parser = StrOutputParser()

# prompt1 = PromptTemplate(
#     template="write a summary for the following poem: {poem}",
#     input_variables=["poem"]
# )

loader = PyPDFLoader('COI_UNIT_1.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size  = 100,
    chunk_overlap = 0,
    separator=''
)

result = splitter.split_documents(docs)

print(result[1].page_content)