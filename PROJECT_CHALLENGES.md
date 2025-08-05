# ⚡ Challenges Faced in LexiAUS (LoRA AustLII Project)

This document lists the **major challenges** we encountered while building the **LexiAUS – Australian Legal Assistant** project, from dataset preparation to LoRA training and app deployment.

---

## 1️⃣ Dataset Preparation Challenges

1. **RTF/PDF to JSON Conversion**
   - Original documents from AustLII were in **RTF** format.
   - Needed to convert them into **plain text** and then **JSON** for LoRA fine-tuning.
   - Faced multiple library issues with `pypandoc` and `pandoc` installation on Windows.

2. **Mixed File Formats**
   - Some documents were **PDFs**, some were **RTFs**, which required:
     - PDF parsing with `PyPDF2` / `pdfplumber`
     - RTF parsing with `pypandoc`
   - Needed a **unified pipeline** to handle both formats.

3. **Text Chunking**
   - Prepared two datasets:
     - `lora_dataset.json` for LoRA instruction-tuning.
     - `rag_chunks.json` for FAISS-based Retrieval-Augmented Generation (RAG).
   - Had to manage **chunk size and overlap** to balance model context.

---

## 2️⃣ LoRA Training Challenges

1. **Model Selection and VRAM Limits**
   - Started with **LLaMA 7B / Mistral 7B**, but **8GB GPU was insufficient**.
   - Faced errors like:
     - `OSError: Paging file is too small`
     - `RuntimeError: Meta tensors cannot be moved`
   - Final solution: **OpenLLaMA 3B in 4-bit quantization (QLoRA)**.

2. **BitsAndBytes & Quantization Setup**
   - Encountered issues with:
     - `bitsandbytes` on native Windows.
     - Needed `load_in_4bit=True` and `bnb_4bit_quant_type="nf4"`.
   - Solved by **using UV Python and careful environment setup**.

3. **Gradient Checkpointing & Training Stability**
   - Initial runs caused:
     - `element 0 of tensors does not require grad` error
   - Resolved by:
     - Using `model.gradient_checkpointing_enable()`
     - Ensuring correct dataset formatting for LoRA.

4. **LoRA Output Quality**
   - Initial outputs:
     - **Repeated instructions**
     - **Looped responses** (e.g., repeated Force Majeure example)
   - Improved with:
     - Correct **instruction-style prompts**
     - `temperature`, `top_p`, and `repetition_penalty` tuning.

---

## 3️⃣ Inference & Deployment Challenges

1. **Device Map & Offloading**
   - Faced multiple errors:
     - `We need an offload_dir to dispatch this model`
     - `RuntimeError: You can't move a model that has some modules offloaded`
   - Learned:
     - **Never call `.to("cuda")`** with `device_map="auto"`
     - Always set `offload_folder` when offloading.

2. **Quantized Inference for 8GB GPU**
   - Switched to **QLoRA (4-bit)** to avoid disk offloading.
   - Increased performance and allowed **full GPU inference**.

3. **Gradio App Development**
   - Built two versions:
     1. **LoRA-only chat app** (pure model responses)
     2. **LoRA + RAG app** (document-grounded responses)
   - Added:
     - **Temperature & top_p** sampling for human-like answers.
     - Clear **instruction prompts** for legal summarization.

4. **Ignored Generation Flags**
   - Warning:
     ```
     The following generation flags are not valid: ['temperature', 'top_p']
     ```
   - Resolved by using `do_sample=True` in `generate()`.

---

## 4️⃣ Project Management Challenges

1. **Large File Handling**
   - AustLII documents + FAISS DB + LoRA weights took **several GB**.
   - Needed **disk cleanup and offload folders**.

2. **Windows Environment Limitations**
   - Some libraries (e.g., `bitsandbytes`) are **Linux-optimized**.
   - Encountered **meta tensor** and **paging file** issues.
   - Considered **WSL2 or Linux** for future training.

3. **Final Achievements**
   - Successfully:
     - Prepared AustLII dataset for LoRA and RAG.
     - Fine-tuned LoRA adapter on **OpenLLaMA 3B**.
     - Built **Gradio Web App (LexiAUS)**.
     - Deployed **quantized model for 8GB GPU**.

---

## ✅ Key Learnings

- **Always check VRAM usage** before choosing the base model.
- **Use 4-bit quantization** for small GPUs.
- **Instruction-style fine-tuning** drastically improves response quality.
- **Never `.to("cuda")` a model with `device_map="auto"`**.
- **RAG improves reliability**, but LoRA-only is faster for pure Q&A.

---

**LexiAUS is now fully functional! 🎉**  
We turned all these challenges into a **working LoRA-powered legal assistant**.
