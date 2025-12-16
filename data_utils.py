import torch
import networkx as nx
from torch_geometric.data import Data

FEATURE_NAMES = [
    "out_txn_count",
    "in_txn_count",
    "out_amount_sum",
    "in_amount_sum",
    "out_amount_avg",
    "in_amount_avg",
    "max_out_amount",
    "balance_change_orig",
    "balance_change_dest",
    "txn_frequency"
]

def build_pyg_graph(df, scaler=None, fit_scaler=False):
    G = nx.DiGraph()

    for _, r in df.iterrows():
        G.add_edge(r.nameOrig, r.nameDest)

    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}

    features, labels = [], []

    for node in nodes:
        out_df = df[df.nameOrig == node]
        in_df = df[df.nameDest == node]

        feat = [
            len(out_df),
            len(in_df),
            out_df.amount.sum(),
            in_df.amount.sum(),
            out_df.amount.mean() if len(out_df) else 0,
            in_df.amount.mean() if len(in_df) else 0,
            out_df.amount.max() if len(out_df) else 0,
            (out_df.oldbalanceOrg - out_df.newbalanceOrig).sum(),
            (in_df.oldbalanceDest - in_df.newbalanceDest).sum(),
            len(out_df) / (df.step.max() + 1)
        ]

        features.append(feat)

        if "isFraud" in df.columns:
            labels.append(
                int(df[(df.nameOrig == node) | (df.nameDest == node)]
                    .isFraud.max())
            )

    X = torch.tensor(features, dtype=torch.float)

    if scaler:
        X = torch.tensor(
            scaler.fit_transform(X) if fit_scaler else scaler.transform(X),
            dtype=torch.float
        )

    edge_index = torch.tensor(
        [(node_map[u], node_map[v]) for u, v in G.edges()],
        dtype=torch.long
    ).t().contiguous()

    y = torch.tensor(labels) if labels else None

    return Data(x=X, edge_index=edge_index, y=y), nodes
