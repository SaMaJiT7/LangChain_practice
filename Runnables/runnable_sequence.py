from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Optional,Literal

load_dotenv()

model = ChatGoogleGenerativeAI(
    model ="gemini-2.5-flash",
    temperature=1.7
)

class JokeResponse(BaseModel):
    joke: str = Field(description="The generated Dark Humour Joke by the LLM.")
    Sensitivity: Literal["Highly sensitive","Moderately sensitive","Less sensitive","Cancellable"] = Field(description="Sensitivity level of the joke according to the People.")

parser = PydanticOutputParser(pydantic_object=JokeResponse)

prompt = PromptTemplate(
    template='Generate a Dark Humour Joke about {topic}.\n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions' : parser.get_format_instructions()} # type: ignore
)
explainprompt = PromptTemplate(
    template='Explain the followinng joke you generated: {joke} in short manner, not more than 50 words.',
    input_variables=['joke']# type: ignore
)
strparser = StrOutputParser()

chains = RunnableSequence(prompt,model,parser)
full_chain = RunnableSequence(explainprompt,model,strparser)

result = chains.invoke({'topic': 'An Indian Person approaching a white girl for dating'})
explaination = full_chain.invoke({'joke': result.joke })

print("The Joke is: ", result.joke)
print("Sensitivity Level: ", result.Sensitivity)

print("\nJoke Explanation: ")
print(explaination)