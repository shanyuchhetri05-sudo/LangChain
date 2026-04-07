from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="CohereLabs/tiny-aya-global",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100}
)

model  = ChatHuggingFace(llm=llm)

model.invoke("what is the captial of switzerdland")