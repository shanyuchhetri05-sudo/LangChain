from langchain_chroma import Chroma
from langchain_core.documents import Document 
from langchain_huggingface import HuggingFaceEmbeddings


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain_core.documents import Document

docs = [
    Document(
        page_content="India is a country located in the southern part of the Asian continent.",
        metadata={"source": "geo", "id": 1}
    ),
    Document(
        page_content="India is the most populated country and has a very diverse culture, languages, and traditions.",
        metadata={"source": "demographics", "id": 2}
    ),
    Document(
        page_content="India's capital is New Delhi. It has 28 states and 8 union territories.",
        metadata={"source": "political", "id": 3}
    ),
    Document(
        page_content="India shares borders with Pakistan, China, Nepal, Bhutan, Bangladesh, and Myanmar.",
        metadata={"source": "geography", "id": 4}
    ),
    Document(
        page_content="India has a mixed economy and is one of the fastest growing major economies in the world.",
        metadata={"source": "economy", "id": 5}
    ),
]

vector_Store = Chroma(embedding_function=embedding,persist_directory='chroma_db',collection_name='sample')

vector_Store.add_documents(docs)

retriver = vector_Store.as_retriever(
    search_type = "mmr",
    search_kwargs={"k":2,"lambda_mult":1}
    )

query = 'what is the capital of india?'
result = retriver.invoke(query)

print(result)