import torch

def calculate_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_size = total_params * 4 / (1024**2)  # Convert to M
    trainable_size = trainable_params * 4 / (1024**2)  # Convert to M

    return total_size, trainable_size