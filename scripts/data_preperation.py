import os 
from PyPDF2 import PdfReader
import json
from striprtf.striprtf import rtf_to_text
from langchain.text_splitter import RecursiveCharacterTextSplitter

INPUT_DIR = "data/Austlii_data/2024"
OUTPUT_DIR= "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rag_chunks=[]
lora_dataset=[]

def extract_text(file_path):
    """Extract text from PDF and RTF file"""
    ext = file_path.lower().split('.')[-1]

    if ext == "pdf":
        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    
    elif ext == "rtf":
        with open(file_path, 'r' , encoding = 'utf-8', errors = 'ignore') as f:
            rtf_content = f.read()
        return  rtf_to_text(rtf_content)
            
    return ""

def chunk_text(text, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_text(text)

#Process all files
for file_name in os.listdir(INPUT_DIR):
    if file_name.lower().endswith((".pdf" , ".rtf")):
        file_path = os.path.join(INPUT_DIR , file_name)
        text=extract_text(file_path)

        if not text.strip():
            continue  #skip empty

        chunks = chunk_text(text)

        # Prepare RAG chunks
        for chunk in chunks:
            rag_chunks.append({"text": chunk, "source": file_name})
            
        #Prepare LoRA dataset (Instruction-Output format)
        for chunk in chunks:
            lora_dataset.append({
                "instruction": "Summarize the following legal text:",
                "input" : chunk,
                "output":"Domain-specific summary to be generate here."
            })

#Save dataset
with open(os.path.join(OUTPUT_DIR, "rag_chunks.json"), "w", encoding = "utf-8") as f:
    json.dump(rag_chunks, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "lora_dataset.json"), "w", encoding ="utf-8") as f:
    json.dump(lora_dataset, f, indent=2, ensure_ascii=False)

print(f"Processed {len(rag_chunks)} chunks for RAG and {len(lora_dataset)} LoRA entries.") 