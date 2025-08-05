import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

#Load chunks
with open("data/processed/rag_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["text"] for item in data]
metadatas = [{"source": item["source"]} for item in data]


#create embeddings

# Use local Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


#Build vector DB
db = FAISS.from_texts(texts,  embeddings, metadatas)

#save for later use
db.save_local("vector_store/faiss_index")
print("Vector DB created and saved")