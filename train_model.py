import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from model import FraudGNN
from data_utils import build_pyg_graph

df = pd.read_csv("fraud_data_balanced_10k.csv")

scaler = StandardScaler()
data, _ = build_pyg_graph(df, scaler=scaler, fit_scaler=True)

idx = torch.arange(data.x.shape[0])
train_idx, test_idx = train_test_split(
    idx, test_size=0.3, stratify=data.y, random_state=42
)

model = FraudGNN(data.x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]))

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = loss_fn(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "fraud_gnn.pt")
joblib.dump(scaler, "scaler.pkl")

print("✅ Training complete.")
