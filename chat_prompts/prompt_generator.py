from langchain_core.prompts import ChatPromptTemplate

# CORRECT WAY: Use tuples ("role", "content")
chat_template = ChatPromptTemplate.from_messages([
    ("human", "Where is my stuff?? Order #5544 was supposed to be here yesterday. Where is it right now?"),
    ("ai", "I apologize for the delay with order #5544. The tracking indicates it is currently held at the local courier facility due to high volume, but it is prioritized for delivery tomorrow morning.")
])

# Now this will work
chat_template.save("customer_support_prompt.json")

print("Template saved successfully!")