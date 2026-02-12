from langchain_community.llms import Ollama
import os

ollama_url = os.getenv("OLLAMA_URL") 

llm = Ollama(
    model="mistral:latest",
    base_url="http://ollama:11434"
)

llm_response = llm.invoke("Bonjour")

print(llm_response)
