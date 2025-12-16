import streamlit as st
import pandas as pd
import torch
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import joblib
import numpy as np
from collections import deque
from sklearn.metrics import confusion_matrix, roc_curve, auc
from model import FraudGNN
from data_utils import build_pyg_graph, FEATURE_NAMES

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="GNN Fraud Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_artifacts():
    model = FraudGNN(len(FEATURE_NAMES))
    model.load_state_dict(torch.load("fraud_gnn.pt", map_location="cpu"))
    model.eval()
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

st.title("🔍 GNN-Based Fraud Detection System")
st.caption("Graph Neural Network powered fraud detection with explainable analytics")

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Controls")

fraud_threshold = st.sidebar.slider(
    "Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.05
)

max_chain_depth = st.sidebar.slider(
    "Fraud Chain Depth", 2, 6, 4
)

uploaded = st.sidebar.file_uploader(
    "Upload Transaction CSV", type=["csv"]
)

if not uploaded:
    st.info("⬅️ Upload a CSV file to begin")
    st.stop()

# ================= LOAD DATA =================
df = pd.read_csv(uploaded).dropna().drop_duplicates()

REQUIRED_COLS = {
    "step","type","amount",
    "nameOrig","nameDest",
    "oldbalanceOrg","newbalanceOrig",
    "oldbalanceDest","newbalanceDest"
}

if not REQUIRED_COLS.issubset(df.columns):
    st.error("CSV schema mismatch.")
    st.stop()

# ================= INFERENCE =================
data, nodes = build_pyg_graph(df, scaler=scaler)

with torch.no_grad():
    probs = torch.softmax(
        model(data.x, data.edge_index), dim=1
    )[:, 1].numpy()

results = pd.DataFrame({
    "account": nodes,
    "fraud_probability": probs
})

results["risk"] = pd.cut(
    results.fraud_probability,
    bins=[0, 0.4, 0.7, 1.0],
    labels=["LOW", "MEDIUM", "HIGH"]
)

fraud_accounts = results[results.fraud_probability >= fraud_threshold]
fraud_set = set(fraud_accounts.account)

# ================= GRAPH =================
G = nx.DiGraph()
for _, r in df.iterrows():
    G.add_edge(r.nameOrig, r.nameDest)

# ================= FRAUD CHAINS =================
def find_fraud_chains(G, fraud_nodes, max_depth):
    chains = []
    visited = set()

    for start in fraud_nodes:
        queue = deque([(start, [start])])
        while queue:
            cur, path = queue.popleft()
            if len(path) > max_depth:
                continue
            for nxt in G.successors(cur):
                new_path = path + [nxt]
                key = tuple(new_path)
                if key in visited:
                    continue
                visited.add(key)

                fraud_count = sum(n in fraud_nodes for n in new_path)
                if fraud_count >= 2:
                    chains.append({
                        "chain": new_path,
                        "length": len(new_path),
                        "fraud_nodes": fraud_count,
                        "risk_score": fraud_count / len(new_path)
                    })
                queue.append((nxt, new_path))
    return sorted(chains, key=lambda x: x["risk_score"], reverse=True)

fraud_chains = find_fraud_chains(G, fraud_set, max_chain_depth)

# ================= KPIs =================
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", len(df))
c2.metric("Accounts", G.number_of_nodes())
c3.metric("Fraud Accounts", len(fraud_accounts))
c4.metric("Fraud Chains", len(fraud_chains))

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🕸️ Network",
    "🎯 Fraud Accounts",
    "⛓️ Fraud Chains",
    "📈 Analytics"
])

# ================= TAB 1: OVERVIEW =================
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.pie(df, names="type", title="Transaction Types"),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            px.histogram(df, x="amount", nbins=50,
                         title="Transaction Amount Distribution"),
            use_container_width=True
        )

    st.subheader("Sample Transactions")
    st.dataframe(df.head(20), use_container_width=True)

# ================= TAB 2: NETWORK (OPTIMIZED) =================
with tab2:
    st.subheader("Transaction Network (Capped for Performance)")

    MAX_NODES = 60
    nodes_vis = list(G.nodes())[:MAX_NODES]
    G_sub = G.subgraph(nodes_vis)

    pos = nx.spring_layout(G_sub, seed=42, k=2, iterations=20)

    edge_x, edge_y = [], []
    for u, v in G_sub.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, node_color, node_text = [], [], [], []
    for n in G_sub.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        is_fraud = n in fraud_set
        node_color.append("red" if is_fraud else "#9ecae1")
        node_text.append(
            f"Account: {n}<br>"
            f"Fraud: {'Yes' if is_fraud else 'No'}<br>"
            f"Degree: {G_sub.degree(n)}"
        )

    fig = go.Figure([
        go.Scatter(x=edge_x, y=edge_y, mode="lines",
                   line=dict(width=0.5, color="#999"),
                   hoverinfo="none"),
        go.Scatter(x=node_x, y=node_y, mode="markers",
                   hoverinfo="text",
                   text=node_text,
                   marker=dict(size=9, color=node_color))
    ])

    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3: FRAUD ACCOUNTS =================
with tab3:
    st.plotly_chart(
        px.histogram(
            fraud_accounts, x="fraud_probability", nbins=30,
            title="Fraud Probability Distribution"
        ),
        use_container_width=True
    )

    risk_counts = (
        fraud_accounts["risk"]
        .value_counts()
        .reset_index(name="count")
        .rename(columns={"index": "risk"})
    )

    st.plotly_chart(
        px.bar(risk_counts, x="risk", y="count",
               title="Risk Category Breakdown"),
        use_container_width=True
    )

    st.dataframe(fraud_accounts, use_container_width=True)

    st.download_button(
        "Download Fraud Accounts CSV",
        fraud_accounts.to_csv(index=False),
        "fraud_accounts.csv"
    )

# ================= TAB 4: FRAUD CHAINS =================
with tab4:
    if not fraud_chains:
        st.info("No fraud chains detected.")
    else:
        for i, c in enumerate(fraud_chains[:15]):
            with st.expander(
                f"Chain {i+1} | Risk {c['risk_score']:.2f}"
            ):
                st.code(" → ".join(c["chain"]))
                st.write(
                    f"{c['fraud_nodes']} of {c['length']} "
                    "accounts are fraudulent."
                )

# ================= TAB 5: ANALYTICS =================
with tab5:
    activity = (
        df.nameOrig.value_counts()
        .add(df.nameDest.value_counts(), fill_value=0)
        .sort_values(ascending=False)
        .head(10)
    )

    st.plotly_chart(
        px.bar(activity, title="Top 10 Active Accounts"),
        use_container_width=True
    )

    if "isFraud" in df.columns:
        y_true = data.y.numpy()
        y_pred = (probs >= fraud_threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        st.plotly_chart(
            px.imshow(cm, text_auto=True,
                      title="Confusion Matrix"),
            use_container_width=True
        )

        fpr, tpr, _ = roc_curve(y_true, probs)
        st.plotly_chart(
            px.area(
                x=fpr, y=tpr,
                title=f"ROC Curve (AUC = {auc(fpr, tpr):.3f})"
            ),
            use_container_width=True
        )

        corr = pd.DataFrame(
            data.x.numpy(), columns=FEATURE_NAMES
        ).corr()

        st.plotly_chart(
            px.imshow(
                corr,
                title="Feature Correlation Matrix",
                zmin=-1, zmax=1,
                color_continuous_scale="RdBu"
            ),
            use_container_width=True
        )
