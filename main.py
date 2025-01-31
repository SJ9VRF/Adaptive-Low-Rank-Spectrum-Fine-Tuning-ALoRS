# To load and use the fine-tuned model:

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load fine-tuned model
model_path = "output/alors_finetuned.pth"
model_name = "bert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Perform inference
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
    return "Positive" if predictions.item() == 1 else "Negative"

# Example usage
text = "This movie was absolutely amazing!"
print(f"Prediction: {predict(text)}")
