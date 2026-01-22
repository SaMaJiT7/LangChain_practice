from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import BraveSearch


search_tool = DuckDuckGoSearchRun()
brave_search = BraveSearch()


result = search_tool.invoke("What is the most controversial thing going on in the world right now?")
# result  = brave_search.invoke("What is the most controversial thing going on in the world right now?")
print(result)
