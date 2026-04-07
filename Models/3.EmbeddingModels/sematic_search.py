from langchain_huggingface import HuggingFaceEmbeddings
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

document = [
    "delhi is the capital of india",
    "kolkata is the capital of west bengal",
    "paris is the capital of france"
]

vector = embedding.embed_documents(document)

query = input(str("enter the query: "))

E_query = embedding.embed_query(query)

scores = cosine_similarity([E_query],vector)[0]


index , score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(document[index])
print(f"similarity score is :{score}")



    
