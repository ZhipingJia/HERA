import torch
import torch.nn.functional as F


def extract_logits(model, dataloader, device):
    model.eval()
    all_logits = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_logits.append(outputs.cpu())

    return torch.cat(all_logits, dim=0)


def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


def calculate_kl_divergence(logits_p, logits_q):
    p = F.softmax(logits_p, dim=-1)
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    return torch.sum(p * (log_p - log_q), dim=-1).mean().item()
