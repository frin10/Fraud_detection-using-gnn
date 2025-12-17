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

def find_fraud_chains(G, fraud_nodes, max_depth=6):
    """
    Enhanced fraud chain detection focusing on 3-4+ fraud account chains
    """
    chains = []
    visited = set()
    fraud_nodes = set(fraud_nodes)

    # Strategy 1: Deep BFS for longer fraud chains (3-4+ fraud accounts)
    for start in fraud_nodes:
        if start not in G:
            continue
            
        queue = deque([(start, [start], 0)])
        
        while queue:
            cur, path, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
                
            try:
                successors = list(G.successors(cur))
            except:
                continue
                
            for nxt in successors:
                new_path = path + [nxt]
                new_depth = depth + 1
                key = tuple(new_path)
                
                if key in visited:
                    continue
                visited.add(key)

                fraud_count = sum(n in fraud_nodes for n in new_path)
                
                # Priority: chains with 3+ fraud accounts
                if fraud_count >= 3:
                    chains.append({
                        "chain": new_path,
                        "length": len(new_path),
                        "fraud_nodes": fraud_count,
                        "risk_score": fraud_count / len(new_path),
                        "type": "sequential",
                        "priority": 1  # High priority
                    })
                # Also accept 2 fraud nodes if chain is longer
                elif fraud_count >= 2 and len(new_path) >= 3:
                    chains.append({
                        "chain": new_path,
                        "length": len(new_path),
                        "fraud_nodes": fraud_count,
                        "risk_score": fraud_count / len(new_path),
                        "type": "sequential",
                        "priority": 2  # Medium priority
                    })
                
                if new_depth < max_depth:
                    queue.append((nxt, new_path, new_depth))
    
    # Strategy 2: Find extended fraud paths (focus on 3+ fraud nodes)
    fraud_list = list(fraud_nodes)[:30]
    for i, start_fraud in enumerate(fraud_list):
        for end_fraud in fraud_list[i+1:]:
            try:
                if nx.has_path(G, start_fraud, end_fraud):
                    # Get all simple paths up to max_depth
                    all_paths = list(nx.all_simple_paths(G, start_fraud, end_fraud, cutoff=max_depth))
                    
                    for path in all_paths[:5]:  # Limit paths per pair
                        path_key = tuple(path)
                        if path_key in visited:
                            continue
                        visited.add(path_key)
                        
                        fraud_in_path = sum(n in fraud_nodes for n in path)
                        if fraud_in_path >= 3:  # Focus on 3+ fraud
                            chains.append({
                                "chain": path,
                                "length": len(path),
                                "fraud_nodes": fraud_in_path,
                                "risk_score": fraud_in_path / len(path),
                                "type": "bridge",
                                "priority": 1
                            })
                        elif fraud_in_path >= 2 and len(path) >= 4:
                            chains.append({
                                "chain": path,
                                "length": len(path),
                                "fraud_nodes": fraud_in_path,
                                "risk_score": fraud_in_path / len(path),
                                "type": "bridge",
                                "priority": 2
                            })
            except:
                continue
    
    # Strategy 3: Fraud clusters (groups of 3+ fraud nodes)
    fraud_subgraph = G.subgraph(fraud_nodes)
    if fraud_subgraph.number_of_nodes() >= 3:
        if hasattr(nx, 'weakly_connected_components'):
            components = list(nx.weakly_connected_components(fraud_subgraph))
        else:
            components = list(nx.connected_components(fraud_subgraph.to_undirected()))
        
        for comp in components:
            if len(comp) >= 3:  # Only clusters with 3+ fraud nodes
                comp_list = list(comp)
                chains.append({
                    "chain": comp_list,
                    "length": len(comp_list),
                    "fraud_nodes": len(comp_list),
                    "risk_score": 1.0,
                    "type": "cluster",
                    "priority": 1
                })
    
    # Strategy 4: Circular patterns with 3+ fraud nodes
    try:
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles[:200]:
            if 3 <= len(cycle) <= max_depth:
                fraud_in_cycle = sum(n in fraud_nodes for n in cycle)
                if fraud_in_cycle >= 3:  # Focus on 3+ fraud in cycles
                    cycle_key = tuple(sorted(cycle))
                    if cycle_key not in visited:
                        visited.add(cycle_key)
                        chains.append({
                            "chain": cycle + [cycle[0]],
                            "length": len(cycle),
                            "fraud_nodes": fraud_in_cycle,
                            "risk_score": fraud_in_cycle / len(cycle),
                            "type": "circular",
                            "priority": 1
                        })
                elif fraud_in_cycle >= 2 and len(cycle) >= 4:
                    cycle_key = tuple(sorted(cycle))
                    if cycle_key not in visited:
                        visited.add(cycle_key)
                        chains.append({
                            "chain": cycle + [cycle[0]],
                            "length": len(cycle),
                            "fraud_nodes": fraud_in_cycle,
                            "risk_score": fraud_in_cycle / len(cycle),
                            "type": "circular",
                            "priority": 2
                        })
    except:
        pass
    
    # Strategy 5: Multi-fraud star patterns (3+ fraud connections)
    for node in list(G.nodes())[:100]:
        if node in fraud_nodes:
            continue
        
        out_fraud = [n for n in G.successors(node) if n in fraud_nodes]
        if len(out_fraud) >= 3:  # 3+ fraud connections
            star_chain = [node] + out_fraud
            chains.append({
                "chain": star_chain,
                "length": len(star_chain),
                "fraud_nodes": len(out_fraud),
                "risk_score": len(out_fraud) / len(star_chain),
                "type": "star_out",
                "priority": 1
            })
        
        in_fraud = [n for n in G.predecessors(node) if n in fraud_nodes]
        if len(in_fraud) >= 3:  # 3+ fraud connections
            star_chain = in_fraud + [node]
            chains.append({
                "chain": star_chain,
                "length": len(star_chain),
                "fraud_nodes": len(in_fraud),
                "risk_score": len(in_fraud) / len(star_chain),
                "type": "star_in",
                "priority": 1
            })
    
    # Remove duplicates
    unique_chains = []
    seen_chains = set()
    
    for chain in chains:
        chain_set = frozenset(chain["chain"])
        if chain_set not in seen_chains:
            seen_chains.add(chain_set)
            unique_chains.append(chain)
    
    # Sort by priority, then risk score, then fraud count
    return sorted(unique_chains, 
                  key=lambda x: (x.get("priority", 3), x["risk_score"], x["fraud_nodes"]), 
                  reverse=False)  # Lower priority number = higher priority

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
    
    chains = current_data['fraud_chains'][:50]
    df = current_data['df']
    results = current_data['results']
    G = current_data['G']
    
    # Format chains with detailed account info
    formatted_chains = []
    for i, c in enumerate(chains):
        # Get detailed info for each account in the chain
        account_details = []
        for account in c['chain']:
            # Get fraud probability
            acc_result = results[results.account == account]
            fraud_prob = float(acc_result['fraud_probability'].iloc[0]) if not acc_result.empty else 0.0
            risk_level = str(acc_result['risk'].iloc[0]) if not acc_result.empty else 'UNKNOWN'
            is_fraud = fraud_prob >= 0.5
            
            # Get transaction stats
            as_orig = df[df.nameOrig == account]
            as_dest = df[df.nameDest == account]
            
            total_sent = float(as_orig['amount'].sum()) if not as_orig.empty else 0.0
            total_received = float(as_dest['amount'].sum()) if not as_dest.empty else 0.0
            num_txns = len(as_orig) + len(as_dest)
            
            # Get connections
            out_degree = G.out_degree(account) if account in G else 0
            in_degree = G.in_degree(account) if account in G else 0
            
            account_details.append({
                'id': account,
                'fraud_prob': fraud_prob,
                'risk_level': risk_level,
                'is_fraud': is_fraud,
                'total_sent': total_sent,
                'total_received': total_received,
                'num_transactions': num_txns,
                'out_degree': out_degree,
                'in_degree': in_degree
            })
        
        # Calculate chain transaction flow
        chain_amount = 0.0
        for j in range(len(c['chain']) - 1):
            src = c['chain'][j]
            dest = c['chain'][j + 1]
            edge_txns = df[(df.nameOrig == src) & (df.nameDest == dest)]
            chain_amount += float(edge_txns['amount'].sum()) if not edge_txns.empty else 0.0
        
        formatted_chains.append({
            'id': i + 1,
            'chain': c['chain'],
            'chain_str': ' → '.join([acc[:8] + '...' for acc in c['chain']]),
            'length': c['length'],
            'fraud_nodes': c['fraud_nodes'],
            'risk_score': c['risk_score'],
            'type': c.get('type', 'sequential'),
            'priority': c.get('priority', 3),
            'total_amount': chain_amount,
            'account_details': account_details
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
    import sys
    
    # Check if running from Streamlit
    if 'streamlit' in sys.modules:
        print("⚠️ WARNING: This is a Flask application, not a Streamlit app!")
        print("Please run it directly with: python app.py")
        print("Do not use: streamlit run app.py")
        sys.exit(1)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)