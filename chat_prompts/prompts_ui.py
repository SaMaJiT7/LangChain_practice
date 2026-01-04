from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt
import json



load_dotenv()

st.header("Summarization Tool")
# Debug: Check if token is actually loaded
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    print("CRITICAL ERROR: HUGGINGFACEHUB_API_TOKEN not found.")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation",  # <--- FIXED: Use hyphen!
    do_sample=False,
    repetition_penalty=1.03,
) # type: ignore

model = ChatHuggingFace(llm=llm)






paper= st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "Pre-trained Language Models for the Legal Domain: A Case Study on Indian Law", "Diffusion Models Beat GANs on Image Synthesis", "Improved pulmonary embolism detection in CT pulmonary angiogram scans with hybrid vision transformers and deep learning techniques"] )

styles = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] )

lengths = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt("summarization_prompt.json")




if st.button("Summarize"):
    chain = template | model
    model_response = chain.invoke({
    'paper_input': paper,
    'style_input': styles,
    'length_input': lengths
    })
    st.write("The summary is ready!")
    st.text_area("Summary:", value=model_response.content, height=200)