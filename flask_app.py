# app.py - Main Flask Application
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import torch
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import joblib
import numpy as np
import json
import io
from collections import deque
from sklearn.metrics import confusion_matrix, roc_curve, auc
from werkzeug.utils import secure_filename
import os

try:
    from model import FraudGNN
    from data_utils import build_pyg_graph, FEATURE_NAMES
except ImportError:
    print("Warning: model.py or data_utils.py not found")
    FEATURE_NAMES = []

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.secret_key = 'your-secret-key-here'

# Create uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store model and data
model = None
scaler = None
current_data = None

# ================= LOAD MODEL =================
def load_model():
    global model, scaler
    try:
        model = FraudGNN(len(FEATURE_NAMES))
        model.load_state_dict(torch.load("fraud_gnn.pt", map_location="cpu"))
        model.eval()
        scaler = joblib.load("scaler.pkl")
        return True, "Model loaded successfully"
    except Exception as e:
        return False, f"Error loading model: {str(e)}"

# Load model on startup
load_model()

# ================= HELPER FUNCTIONS =================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def find_fraud_chains(G, fraud_nodes, max_depth=4):
    chains = []
    visited = set()

    for start in fraud_nodes:
        if start not in G:
            continue
            
        queue = deque([(start, [start])])
        
        while queue:
            cur, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
                
            try:
                successors = list(G.successors(cur))
            except:
                continue
                
            for nxt in successors:
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

def process_data(df, fraud_threshold=0.5):
    """Process uploaded data and return results"""
    global current_data
    
    # Required columns
    REQUIRED_COLS = {
        "step", "type", "amount",
        "nameOrig", "nameDest",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest"
    }
    
    if not REQUIRED_COLS.issubset(df.columns):
        return None, f"Missing required columns: {REQUIRED_COLS - set(df.columns)}"
    
    # Build graph and get predictions
    try:
        data, nodes = build_pyg_graph(df, scaler=scaler)
        
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        
        results = pd.DataFrame({
            "account": nodes,
            "fraud_probability": probs
        })
        
        results["risk"] = pd.cut(
            results.fraud_probability,
            bins=[0, 0.4, 0.7, 1.0],
            labels=["LOW", "MEDIUM", "HIGH"]
        )
        
        fraud_accounts = results[results.fraud_probability >= fraud_threshold].copy()
        fraud_set = set(fraud_accounts.account)
        
        # Build network graph
        G = nx.DiGraph()
        for _, r in df.iterrows():
            G.add_edge(r.nameOrig, r.nameDest)
        
        # Find fraud chains
        fraud_chains = find_fraud_chains(G, fraud_set, max_depth=4)
        
        # Store current data
        current_data = {
            'df': df,
            'results': results,
            'fraud_accounts': fraud_accounts,
            'G': G,
            'fraud_chains': fraud_chains,
            'data': data,
            'probs': probs,
            'nodes': nodes
        }
        
        return current_data, None
        
    except Exception as e:
        return None, f"Error processing data: {str(e)}"

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV files allowed'}), 400
    
    try:
        # Read CSV
        df = pd.read_csv(file).dropna().drop_duplicates()
        
        # Get parameters
        fraud_threshold = float(request.form.get('threshold', 0.5))
        
        # Process data
        result, error = process_data(df, fraud_threshold)
        
        if error:
            return jsonify({'error': error}), 400
        
        # Return summary statistics
        return jsonify({
            'success': True,
            'stats': {
                'total_transactions': len(df),
                'total_accounts': result['G'].number_of_nodes(),
                'fraud_accounts': len(result['fraud_accounts']),
                'fraud_chains': len(result['fraud_chains'])
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/overview')
def get_overview():
    if current_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    df = current_data['df']
    
    # Transaction types
    type_counts = df['type'].value_counts().to_dict()
    
    # Amount distribution
    amounts = df['amount'].tolist()
    
    # Sample transactions
    sample = df.head(20).to_dict('records')
    
    return jsonify({
        'type_counts': type_counts,
        'amounts': amounts,
        'sample_transactions': sample
    })

@app.route('/api/network')
def get_network():
    if current_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    G = current_data['G']
    fraud_set = set(current_data['fraud_accounts']['account'])
    df = current_data['df']
    results = current_data['results']
    
    # Limit nodes for performance
    MAX_NODES = 60
    nodes_vis = list(G.nodes())[:MAX_NODES]
    G_sub = G.subgraph(nodes_vis)
    
    pos = nx.spring_layout(G_sub, seed=42, k=2, iterations=20)
    
    # Build network data with enhanced info
    edges = []
    for u, v in G_sub.edges():
        # Get transaction details for this edge
        edge_txns = df[(df.nameOrig == u) & (df.nameDest == v)]
        total_amount = edge_txns['amount'].sum() if not edge_txns.empty else 0
        txn_count = len(edge_txns)
        
        edges.append({
            'source': u,
            'target': v,
            'x0': pos[u][0],
            'y0': pos[u][1],
            'x1': pos[v][0],
            'y1': pos[v][1],
            'amount': float(total_amount),
            'count': int(txn_count)
        })
    
    nodes = []
    for n in G_sub.nodes():
        # Get fraud probability for this node
        node_result = results[results.account == n]
        fraud_prob = float(node_result['fraud_probability'].iloc[0]) if not node_result.empty else 0.0
        risk_level = str(node_result['risk'].iloc[0]) if not node_result.empty else 'UNKNOWN'
        
        # Get transaction stats - both from full graph and subgraph
        out_degree = G_sub.out_degree(n)
        in_degree = G_sub.in_degree(n)
        total_degree = out_degree + in_degree
        
        # Get account transaction statistics from original dataframe
        as_orig = df[df.nameOrig == n]
        as_dest = df[df.nameDest == n]
        
        total_sent = float(as_orig['amount'].sum()) if not as_orig.empty else 0.0
        total_received = float(as_dest['amount'].sum()) if not as_dest.empty else 0.0
        num_sent = len(as_orig)
        num_received = len(as_dest)
        
        # Get transaction types
        txn_types = list(set(as_orig['type'].tolist() + as_dest['type'].tolist()))
        
        nodes.append({
            'id': n,
            'x': pos[n][0],
            'y': pos[n][1],
            'is_fraud': n in fraud_set,
            'fraud_prob': fraud_prob,
            'risk_level': risk_level,
            'out_degree': out_degree,
            'in_degree': in_degree,
            'total_degree': total_degree,
            'total_sent': total_sent,
            'total_received': total_received,
            'num_sent': num_sent,
            'num_received': num_received,
            'txn_types': ', '.join(txn_types) if txn_types else 'None'
        })
    
    return jsonify({
        'nodes': nodes,
        'edges': edges
    })

@app.route('/api/fraud_accounts')
def get_fraud_accounts():
    if current_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    fraud_accounts = current_data['fraud_accounts']
    
    # Probability distribution
    probs = fraud_accounts['fraud_probability'].tolist()
    
    # Risk breakdown
    risk_counts = fraud_accounts['risk'].value_counts().to_dict()
    
    # Account list
    accounts = fraud_accounts.sort_values('fraud_probability', ascending=False).to_dict('records')
    
    return jsonify({
        'probabilities': probs,
        'risk_counts': risk_counts,
        'accounts': accounts
    })

@app.route('/api/fraud_chains')
def get_fraud_chains():
    if current_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    chains = current_data['fraud_chains'][:15]
    
    # Format chains for JSON
    formatted_chains = []
    for i, c in enumerate(chains):
        formatted_chains.append({
            'id': i + 1,
            'chain': c['chain'],
            'chain_str': ' → '.join(c['chain']),
            'length': c['length'],
            'fraud_nodes': c['fraud_nodes'],
            'risk_score': c['risk_score']
        })
    
    return jsonify({'chains': formatted_chains})

@app.route('/api/analytics')
def get_analytics():
    if current_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    df = current_data['df']
    data = current_data['data']
    probs = current_data['probs']
    
    # Top active accounts
    orig_counts = df.nameOrig.value_counts()
    dest_counts = df.nameDest.value_counts()
    activity = orig_counts.add(dest_counts, fill_value=0).sort_values(ascending=False).head(10)
    
    top_accounts = {
        'accounts': activity.index.tolist(),
        'counts': activity.values.tolist()
    }
    
    result = {
        'top_accounts': top_accounts
    }
    
    # If ground truth available
    if "isFraud" in df.columns:
        y_true = data.y.numpy()
        y_pred = (probs >= 0.5).astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        result['confusion_matrix'] = cm.tolist()
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, probs)
        result['roc'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(auc(fpr, tpr))
        }
        
        # Threshold analysis
        thresholds = np.linspace(0.01, 0.99, 50)
        recalls = []
        precisions = []
        
        for t in thresholds:
            preds = (probs >= t).astype(int)
            tp = ((preds == 1) & (y_true == 1)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            
            recall = tp / (tp + fn + 1e-9)
            precision = tp / (tp + fp + 1e-9)
            
            recalls.append(float(recall))
            precisions.append(float(precision))
        
        result['threshold_analysis'] = {
            'thresholds': thresholds.tolist(),
            'recalls': recalls,
            'precisions': precisions
        }
    
    # Temporal analysis
    if "step" in df.columns:
        results_df = current_data['results']
        step_map = {}
        
        for acc, p in zip(results_df.account, results_df.fraud_probability):
            acc_steps = df[
                (df.nameOrig == acc) | (df.nameDest == acc)
            ]["step"].unique()
            
            for s in acc_steps:
                step_map.setdefault(int(s), []).append(float(p))
        
        fraud_trend = {
            'steps': list(step_map.keys()),
            'avg_probs': [np.mean(v) for v in step_map.values()]
        }
        
        # Sort by step
        sorted_indices = np.argsort(fraud_trend['steps'])
        fraud_trend['steps'] = [fraud_trend['steps'][i] for i in sorted_indices]
        fraud_trend['avg_probs'] = [fraud_trend['avg_probs'][i] for i in sorted_indices]
        
        result['temporal'] = fraud_trend
    
    return jsonify(result)

@app.route('/api/account/<account_id>')
def get_account_transactions(account_id):
    if current_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    df = current_data['df']
    
    # Get transactions involving this account
    txns = df[
        (df.nameOrig == account_id) | (df.nameDest == account_id)
    ].sort_values("step").to_dict('records')
    
    return jsonify({'transactions': txns})

@app.route('/download/fraud_accounts')
def download_fraud_accounts():
    if current_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    fraud_accounts = current_data['fraud_accounts']
    
    # Create CSV in memory
    output = io.StringIO()
    fraud_accounts.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='fraud_accounts.csv'
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)