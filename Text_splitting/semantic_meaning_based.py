from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import json

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
 
document =    """delhi is the capital of india
    kolkata is the capital of west bengal
    paris is the capital of france"""


splitter = SemanticChunker(
    embedding,breakpoint_threshold_amount="standard_deviation",
    breakpoint_threshold_type=1
)

docs = splitter.create_documents(document)

print(len(docs))

print(docs)