from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers.document_compressors import LLMChainExtractor 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import ContextualCompressionRetriever



embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs =[
    Document(page_content='India is country loacted in southern parts of asian contient'),
    Document(page_content='India is a diverse contry it is the most populated country '),
    Document(page_content="India's capital is delhi it has 29 stated and 7 union territories"),
]

vector_Store = Chroma(embedding_function=embedding,persist_directory='chroma_db',collection_name='sample')

vector_Store.add_documents(docs)

llm = ""
compressor = LLMChainExtractor.from_llm(llm)

base_retriver = vector_Store.as_retriever(search_kwargs={"k":2})

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriver,
    base_compressor=compressor

)

query = "what is the capital of india"
result  = compression_retriever.invoke(query)


print(result)