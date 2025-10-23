import torch
from .metric import compute_auc, compute_accuracy

def train_epoch(model, optimizer, criterion, data, device, batch_size=1024):
    model.train()
    total_loss = 0
    for batch_start in range(0, len(data), batch_size):
        batch = data.iloc[batch_start: batch_start+batch_size]
        u = torch.tensor(batch["u"].values, dtype=torch.long, device=device)
        i = torch.tensor(batch["i"].values, dtype=torch.long, device=device)
        y = torch.tensor(batch["label"].values, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        preds = model(u, i)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)
    return total_loss / len(data)

def evaluate(model, data, device, batch_size=1024):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_start in range(0, len(data), batch_size):
            batch = data.iloc[batch_start: batch_start+batch_size]
            u = torch.tensor(batch["u"].values, dtype=torch.long, device=device)
            i = torch.tensor(batch["i"].values, dtype=torch.long, device=device)
            preds = model(u, i).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].values)
    auc = compute_auc(all_labels, all_preds)
    acc = compute_accuracy(all_labels, all_preds)
    return auc, acc
