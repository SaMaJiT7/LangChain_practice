from langchain_community.retrievers import WikipediaRetriever
import urllib3
import warnings

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

# Set environment variable to disable SSL verification
import os
os.environ['PYTHONHTTPSVERIFY'] = '0'

retriever = WikipediaRetriever(top_k_results=2, language="en") # type: ignore

# Use a simpler, more direct query for better results
# Wikipedia works better with specific topic names, not full questions
query = "Syrian Civil War"  # Changed from complex question to specific topic

print("Starting Wikipedia search...")
print(f"Query: {query}\n")

try:
    result = retriever.invoke(query)
    print(f"Found {len(result)} documents\n")
    
    if not result:
        print("No documents found for this query.")
    else:
        for i, doc in enumerate(result):
            print(f"Document {i+1}:\n")
            print(doc.page_content)
            print("\n"+"-"*80+"\n")
            
except Exception as e:
    print(f"Error occurred: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nThis is likely a network/SSL connection issue.")
    print("Possible solutions:")
    print("1. Check your internet connection")
    print("2. Try using a VPN")
    print("3. Check if Wikipedia is accessible from your network")
    print("4. Check firewall/antivirus settings")