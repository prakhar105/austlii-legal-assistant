import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------
# Config
# ---------------------------
BASE_MODEL = "openlm-research/open_llama_3b"   # Base model
LORA_PATH = "lora_model"                       # Your trained LoRA adapter
MAX_NEW_TOKENS = 256                           # Max output length

# ---------------------------
# 1. BitsAndBytes Config for 4-bit Quantization
# ---------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",      # Normalized float 4 (best for QLoRA)
    bnb_4bit_compute_dtype=torch.bfloat16,  # Mixed precision compute
    bnb_4bit_use_double_quant=True  # Extra quantization layer to save memory
)

# ---------------------------
# 2. Load Tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------
# 3. Load Base Model in 4-bit
# ---------------------------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"  # Fully managed by Accelerate
)

# ---------------------------
# 4. Load LoRA Adapter
# ---------------------------
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()  # Inference mode

print("✅ Model and LoRA adapter loaded in 4-bit quantization!")

# ---------------------------
# 5. Simple Inference Example
# ---------------------------
prompt = "Summarize the termination clause of a contract in simple terms."

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

print("\n--- LoRA Quantized Output ---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
