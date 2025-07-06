import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Leonet_model import LeoNet

class LeoNetDataset(Dataset):
    def __init__(self, jsonl_file, vocab=None, seq_len=8):
        self.samples = []
        self.seq_len = seq_len
        self.vocab = vocab if vocab else list("abcdefghijklmnopqrstuvwxyz ")
        with open(jsonl_file, "r") as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        inp = self.samples[idx]["input"].lower()
        tokens = [self.vocab.index(c) if c in self.vocab else len(self.vocab)-1 for c in inp][:self.seq_len]
        tokens = tokens + [0]*(self.seq_len - len(tokens))
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(self.samples[idx]["motor_output"], dtype=torch.float)
        return x, y

# Load the dataset and dataloader
dataset = LeoNetDataset("leonet_fullrange_dataset.jsonl")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model setup
vocab_size = len(dataset.vocab)
model = LeoNet(vocab_size=vocab_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
mse_loss = nn.MSELoss()

# Training loop
epochs = 12
for epoch in range(epochs):
    total_loss = 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits, motor_pred = model(xb)
        loss = mse_loss(motor_pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

# Save the trained model
torch.save(model.state_dict(), "leonet_fullrange_pretrained.pth")
print("Model saved as leonet_fullrange_pretrained.pth")
