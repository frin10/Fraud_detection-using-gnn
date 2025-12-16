import torch
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)

from model import FraudGNN
from data_utils import build_pyg_graph, FEATURE_NAMES

df = pd.read_csv("fraud_data_balanced_10k.csv")

scaler = joblib.load("scaler.pkl")
data, _ = build_pyg_graph(df, scaler=scaler)

model = FraudGNN(data.x.shape[1])
model.load_state_dict(torch.load("fraud_gnn.pt"))
model.eval()

with torch.no_grad():
    logits = model(data.x, data.edge_index)
    probs = torch.softmax(logits, dim=1)[:, 1]
    preds = (probs > 0.5).int()

print(classification_report(data.y, preds))

cm = confusion_matrix(data.y, preds)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, _ = roc_curve(data.y, probs)
plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
plt.legend()
plt.show()

precision, recall, _ = precision_recall_curve(data.y, probs)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

feat_df = pd.DataFrame(data.x.numpy(), columns=FEATURE_NAMES)
sns.heatmap(feat_df.corr(), cmap="coolwarm", center=0)
plt.title("Feature Correlation")
plt.show()
