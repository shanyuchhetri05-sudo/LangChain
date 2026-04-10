from langchain_community.retrievers import WikipediaRetriever

retrivier = WikipediaRetriever(top_k_results=3,lang='en')

query = "tell me about dhurandar movie"

docs = retrivier.invoke(query)

print(docs.summary)