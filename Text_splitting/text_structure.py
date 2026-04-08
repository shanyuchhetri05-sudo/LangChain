from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """""In LangChain, the CharacterTextSplitter is the most basic text splitting utility. It divides text into chunks based on a single, specific character (the separator)" \
and measures the resulting chunk size by the number of character"""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0  )

result = splitter.split_text(text)

print(result)
