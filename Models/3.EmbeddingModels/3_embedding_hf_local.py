from langchain_huggingface import HuggingFaceEmbeddings
import json

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

document = [
    "delhi is the capital of india"
    "kolkata is the capital of west bengal"
    "paris is the capital of france"
]

vector = embedding.embed_documents(document)

json_string = json.dumps(vector)

with open("document.json","w") as f:
    f = f.write(json_string)

