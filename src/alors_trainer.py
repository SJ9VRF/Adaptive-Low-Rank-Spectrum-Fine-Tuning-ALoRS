import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AdamW

class ALoRSTrainer:
    def __init__(self, model, high_snr_layers, mid_snr_layers, train_dataloader, eval_dataloader, output_dir):
        self.model = model
        self.high_snr_layers = high_snr_layers
        self.mid_snr_layers = mid_snr_layers
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Define adaptive LoRA configurations
        high_rank = 16
        mid_rank = 4

        self.lora_config_high = LoraConfig(
            target_modules=self.high_snr_layers,
            r=high_rank,
            lora_alpha=32,
            lora_dropout=0.05
        )

        self.lora_config_mid = LoraConfig(
            target_modules=self.mid_snr_layers,
            r=mid_rank,
            lora_alpha=16,
            lora_dropout=0.05
        )

        # Apply different LoRA ranks to high & mid SNR layers
        self.model = get_peft_model(self.model, self.lora_config_high)
        self.model = get_peft_model(self.model, self.lora_config_mid)

    def train(self, epochs=3, lr=2e-5):
        optimizer = AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            for batch in self.train_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

        torch.save(self.model.state_dict(), f"{self.output_dir}/alors_finetuned.pth")

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy: {100 * correct / total:.2f}%")

