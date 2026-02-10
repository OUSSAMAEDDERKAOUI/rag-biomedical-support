from app.rag.retriever import get_retriever

print(" Initialisation du retriever...")

retriever = get_retriever()

print(" Retriever prêt")

query = "Qu'est-ce que  Balances  ?"

print(f"\n Recherche pour : {query}\n")

docs = retriever.get_relevant_documents(query)

print(f" Nombre de résultats : {len(docs)}\n")

for i, d in enumerate(docs):
    print(f"----- CHUNK {i+1} -----")
    print(d.page_content)
    print("\nMETADATA :", d.metadata)
    print("\n")
