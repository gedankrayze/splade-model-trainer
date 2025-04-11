from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import json

# Path to your trained model
model_dir = "./fine_tuned_splade/wolf-splade-pp-v2"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForMaskedLM.from_pretrained(model_dir)

# Load SPLADE config (saved during training)
with open(f"{model_dir}/splade_config.json", "r") as f:
    splade_config = json.load(f)

# Move to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    print("Using MPS (Metal Performance Shaders) for GPU acceleration.")
    device = torch.device("mps")
elif torch.cuda.is_available():
    print("Using CUDA for GPU acceleration.")
    device = torch.device("cuda")
else:
    print("Using CPU for inference.")
    device = torch.device("cpu")

model.to(device)

print(model.config)
