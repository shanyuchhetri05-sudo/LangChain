from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

python_code = """""# Hello World
print("Hello, World!")

# Variables and Input
name = input("Enter your name: ")
print(f"Hello, {name}!")

# List and Loop
fruits = ["Apple", "Banana", "Cherry"]
for fruit in fruits:
    print(fruit)"""

splitter=RecursiveCharacterTextSplitter.from_language(
    language= Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0  )

result = splitter.split_text(python_code)

print(result)