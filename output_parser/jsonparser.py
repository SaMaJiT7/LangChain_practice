import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# 1. SETUP MODEL
# Switching to Mistral-7B-Instruct-v0.3 (Excellent at JSON)
LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    do_sample=False,
    repetition_penalty=1.03,
    temperature=0.1, # Keep low for structured data
    max_new_tokens=1024
) # type: ignore

model = ChatHuggingFace(llm=LLM)

# 2. SETUP PARSER
parser = JsonOutputParser()

# 3. SETUP TEMPLATE
template = PromptTemplate(
    template="""
    You are a helpful assistant.
    Give me five interesting and niche facts about {topic}.
    
    {Format_Instruction}
    """,
    input_variables=['topic'],
    # CRITICAL FIX: Add parentheses () to call the function
    partial_variables={'Format_Instruction': parser.get_format_instructions()}
)

# 4. RUN CHAIN
chain = template | model | parser

print("Running Chain...")
try:
    final_response = chain.invoke({'topic': 'Narendra Modi'})
    print(final_response)
    
    # Verify it is a real dictionary
    print(f"\nType: {type(final_response)}")
    
except Exception as e:
    print(f"Error: {e}")