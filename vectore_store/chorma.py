from langchain_chroma import Chroma
from langchain_core.documents import Document 
from langchain_huggingface import HuggingFaceEmbeddings


embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

docs = [
    Document(
        page_content="Virat Kohli is one of the greatest batsmen in cricket history. He is known for his consistency, aggressive batting style, and leadership. He has scored thousands of runs in the IPL and international cricket.",
        metadata={"team": "RCB"}
    ),
    Document(
        page_content="MS Dhoni is a legendary wicketkeeper-batsman and captain. He is famous for his calm mindset and finishing ability in pressure situations. He has led his team to multiple IPL titles.",
        metadata={"team": "CSK"}
    ),
    Document(
        page_content="Rohit Sharma is a top-order batsman known for his elegant stroke play and big scores. He has captained his team to several IPL championships and is one of the most successful IPL captains.",
        metadata={"team": "MI"}
    ),
    Document(
        page_content="KL Rahul is a stylish batsman known for his versatility and ability to anchor innings. He consistently performs as a top-order batsman in the IPL.",
        metadata={"team": "LSG"}
    ),
    Document(
        page_content="Hardik Pandya is a dynamic all-rounder known for his explosive batting and useful bowling. He plays a crucial role in finishing matches and contributing in multiple departments.",
        metadata={"team": "GT"}
    )
]

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory='chroma_db',
    collection_name='sample'
)

#store docs
vector_store.add_documents(docs)

#view documents
vector_store.get(include=['embedding','docs','metadatas'])

#similarity search
vector_store.similarity_search(
    query='who among these are a bowler?',
    k=1
)

#search with similaritiy scores
vector_store.similarity_search_with_score(
    query='who among these are a bowler?',
    k=1
)

#meta-data filtering
vector_store.similarity_search_with_scores(
    query = "",
    filter = {'team':'chennai super king'}
)

#update document
updated_doc = Document(
    page_content="""
Virat Kohli is one of the most accomplished modern-day cricketers and a former captain of the Indian national team. Known for his exceptional consistency, fitness, and aggressive mindset, he has been a dominant force in all formats of the game. 

In the Indian Premier League (IPL), Kohli has been a long-time player for Royal Challengers Bangalore (RCB) and is among the highest run-scorers in the tournament’s history. He holds the record for the most runs in a single IPL season (2016), where he scored 973 runs.

Kohli is especially known for his ability to chase targets under pressure and anchor innings while maintaining a high strike rate. His cover drives and shot selection are widely regarded as some of the best in cricket.

Beyond cricket, he is also recognized for his leadership, discipline, and influence on fitness standards in Indian cricket.
""",
    metadata={
        "team": "RCB",
        "role": "Batsman",
        "nationality": "India",
        "ipl_debut": 2008,
        "records": ["Most runs in IPL 2016 season", "Among highest IPL run scorers"]
    }
)

vector_store.update_document(document_id='',document=updated_doc)

#delete document
vector_store.delete(ids=[''])

