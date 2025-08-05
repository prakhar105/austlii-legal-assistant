# 📜 LexiAUS – LoRA-Powered Australian Legal Assistant

LexiAUS is an AI-powered **legal assistant** fine-tuned on **AustLII (Australian Legal Information Institute)** legal documents.  
It can:

- **Summarize legal clauses** in plain English  
- **Explain Australian law** in simple terms  
- **Answer legal questions** interactively via a web app  

This project uses **LoRA (Low-Rank Adaptation)** with **OpenLLaMA 3B** and **4‑bit quantization** to run efficiently on **consumer GPUs (8GB RTX 4060)**.

---

## 🚀 Features

- ✅ **LoRA Fine-Tuning** on AustLII documents  
- ✅ **Quantized 4‑bit model** → fits in 8GB GPU  
- ✅ **Interactive Gradio Web App** (`app.py`)  
- ✅ **Instruction-style prompts** for better legal explanations  
- ✅ **Memory-efficient inference** with BitsAndBytes  

---

## 📸 App Screenshot

Here’s how **LexiAUS** looks in action:

![LexiAUS Screenshot](assets/app_screenshot.png)

*(Make sure your screenshot is saved as `assets/app_screenshot.png` in the repo.)*

---

## 📂 Project Structure

```
lexi-aus-lora/
│
├── assets/                     # Screenshots and images for README
│   └── app_screenshot.png
│
├── data/
│   └── processed/              # Prepared LoRA dataset (JSON)
│
├── lora_model/                 # Saved LoRA adapter weights
│
├── scripts/
│   ├── data_preparation.py     # Converts RTF/PDFs → JSON for LoRA
│   ├── train_lora.py           # Fine-tunes LoRA with OpenLLaMA 3B
│   ├── lora_inference.py       # CLI inference script
│   └── app.py                  # Gradio web app for LoRA inference
│
├── vector_store/               # (Optional) FAISS DB for RAG
│   └── faiss_index/
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/lexi-aus-lora.git
   cd lexi-aus-lora
   ```

2. **Create a virtual environment**
   ```bash
   uv venv
   uv pip install --upgrade pip
   ```

3. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

   Minimum requirements:
   ```
   torch
   transformers
   peft
   bitsandbytes
   accelerate
   gradio
   datasets
   faiss-cpu
   sentence-transformers
   safetensors
   ```

---

## 🏋️‍♂️ Training the LoRA Model

1. **Prepare your dataset**
   - Place your RTF/PDF files in a folder  
   - Run `scripts/data_preparation.py` to generate:
     - `lora_dataset.json` for LoRA  
     - `rag_chunks.json` for optional RAG  

2. **Train the LoRA adapter**
   ```bash
   uv run scripts/train_lora.py
   ```
   - Trains a **LoRA adapter** for OpenLLaMA 3B  
   - Uses **gradient checkpointing + 4‑bit quantization** for 8GB VRAM  
   - Outputs saved to `lora_model/`  

---

## 💬 Running the LoRA Chat App

Launch the **web interface**:

```bash
uv run scripts/app.py
```

- Opens a **Gradio app** in your browser  
- Type a legal question or clause → model responds in **plain English**  

**Example:**
```
Input: Explain the termination clause in this contract.
Output: The contract can be ended by the contractor if they cannot perform due to unforeseen events.
```

---

## ⚡ Quantized Inference

- Model runs in **4‑bit quantization** → fits on RTX 4060 / 8GB GPUs  
- `BitsAndBytes` handles efficient inference  
- Sampling enabled for **natural, non-repetitive answers**

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.  
**LexiAUS does not provide legal advice.**  
Always consult a qualified lawyer for official legal interpretation.

---

## 🌟 Acknowledgements

- [AustLII](https://www.austlii.edu.au/) – Source of legal texts  
- [Hugging Face](https://huggingface.co/) – Transformers, PEFT, and datasets  
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) – 4‑bit quantization  
- [Gradio](https://gradio.app/) – Web UI for AI apps  

---

## ✅ Project Summary

- **Project Name:** `lexi-aus-lora`  
- **App Name:** **LexiAUS – Australian Legal Assistant**  
- **Description:** LoRA‑fine‑tuned OpenLLaMA 3B for summarizing and explaining Australian legal clauses  
