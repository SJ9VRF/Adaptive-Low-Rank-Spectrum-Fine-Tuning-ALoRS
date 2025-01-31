 import torch
import torch.nn as nn

class SNRAnalyzer:
    def __init__(self, model, threshold_low=0.2, threshold_high=0.7):
        self.model = model
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def get_snr(self, layer):
        """Compute Signal-to-Noise Ratio (SNR) for a given layer."""
        if hasattr(layer, 'weight') and layer.weight is not None:
            weights = layer.weight.data
            signal = torch.mean(weights).item()
            noise = torch.std(weights).item()
            return signal / noise if noise != 0 else 0
        return 0

    def get_adaptive_snr_layers(self):
        """Identify layers and classify them based on SNR levels."""
        high_snr_layers = []
        mid_snr_layers = []
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LayerNorm, nn.Embedding)):
                snr = self.get_snr(layer)
                if snr > self.threshold_high:
                    high_snr_layers.append(name)
                elif self.threshold_low < snr <= self.threshold_high:
                    mid_snr_layers.append(name)
        return high_snr_layers, mid_snr_layers

