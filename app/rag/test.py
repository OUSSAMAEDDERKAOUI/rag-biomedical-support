from langchain_community.llms import Ollama
import os

# Récupère l'URL de l'API Ollama depuis la variable d'environnement
ollama_url = os.getenv("OLLAMA_URL")  # ex: "http://ollama:11434/api/chat"

# Initialise le LLM Ollama
llm = Ollama(
    model="mistral:latest",
    base_url="http://ollama:11434"
)

# Test simple : envoyer un message "Bonjour"
llm_response = llm.invoke("Bonjour")

# Affiche la réponse de l'assistant
print(llm_response)
