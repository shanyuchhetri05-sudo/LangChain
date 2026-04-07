from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.messages import AIMessage,HumanMessage,SystemMessage


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

chat_history = [
      SystemMessage(content='you are a helpful assistant name jarvis')
                ]

while True:
    user_prompt = input("You: ")
    chat_history.append(HumanMessage(content=user_prompt))

    if user_prompt == "exit":
            break;

    result = model.invoke(chat_history)

    chat_history.append(AIMessage(content=result.text))
    print("AI: ",result.text)

print(chat_history)