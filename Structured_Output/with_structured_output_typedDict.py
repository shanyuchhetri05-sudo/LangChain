from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

class Review(TypedDict):
    key_themes: Annotated[list[str],"write down all the key themes of the review"]
    productName:Annotated[Optional[str],"name the reviewed product"]
    summary: Annotated[str,"A brief summary of the product"]
    sentiment: Annotated[Literal["positive","negative"],"the sentiment of the review"]
    pros: Annotated[Optional[list[str]],"the pros in points where each pint is an element of the list"]
    cons:  Annotated[Optional[list[str]],"the cons in points where each point is an element of the list"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
A high-quality product review is a detailed, balanced, 
and honest evaluation based on first-hand experience, 
guiding consumers through pros, cons, and personal insights. 
It should go beyond technical specs to focus on practical usage, 
comparing alternatives to aid purchasing decisions. 
Essential elements include original photos, a clear verdict,
and identification of the ideal user.
""")

print(result)

print(result["productName"])

print(result["summary"])

print(result["sentiment"])




