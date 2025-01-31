import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from snr_analyzer import SNRAnalyzer
from alors_trainer import ALoRSTrainer

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("imdb")
train_texts = dataset["train"]["text"][:1000]
train_labels = dataset["train"]["label"][:1000]

# Tokenize data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
train_dataset = torch.utils.data.TensorDataset(train_encodings["input_ids"], torch.tensor(train_labels))
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Apply Adaptive SNR Analysis
snr_analyzer = SNRAnalyzer(model)
high_snr_layers, mid_snr_layers = snr_analyzer.get_adaptive_snr_layers()
print(f"High SNR Layers: {high_snr_layers}")
print(f"Mid SNR Layers: {mid_snr_layers}")

# Train with Adaptive LoRA + Spectrum
trainer = ALoRSTrainer(
    model=model,
    high_snr_layers=high_snr_layers,
    mid_snr_layers=mid_snr_layers,
    train_dataloader=train_dataloader,
    eval_dataloader=None,
    output_dir="./output"
)
trainer.train()
trainer.evaluate()

