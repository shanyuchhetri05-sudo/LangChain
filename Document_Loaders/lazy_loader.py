from langchain_community.document_loaders  import DirectoryLoader,PyPDFLoader,

loader = DirectoryLoader(
    path = '../COI_NOTES',
    glob= '*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

print(len(docs))

print(docs[0].metadata)