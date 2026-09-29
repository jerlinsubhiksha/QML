import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

# Qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from scipy.optimize import minimize

N_QUBITS = 4
N_LAYERS = 2

def quantum_expectations(x, qparams):
    """
    Exact quantum circuit structure used by QADAPT V12.
    """
    qc = QuantumCircuit(N_QUBITS)
    
    # Patient-data encoding
    for q in range(N_QUBITS):
        qc.ry(float(x[q]), q)
        qc.rz(float(0.5 * x[q]), q)

    idx = 0
    # Two trainable layers
    for _ in range(N_LAYERS):
        for q in range(N_QUBITS):
            qc.ry(float(qparams[idx]), q)
            idx += 1
            qc.rz(float(qparams[idx]), q)
            idx += 1

        # Ring entanglement
        for q in range(N_QUBITS - 1):
            qc.cx(q, q + 1)
        qc.cx(N_QUBITS - 1, 0)

        # Data re-uploading
        for q in range(N_QUBITS):
            qc.ry(float(0.5 * x[q]), q)

    state = Statevector.from_instruction(qc)
    output = []
    
    for q in range(N_QUBITS):
        pauli = ["I"] * N_QUBITS
        pauli[N_QUBITS - 1 - q] = "Z"
        observable = SparsePauliOp.from_list([("".join(pauli), 1.0)])
        output.append(float(np.real(state.expectation_value(observable))))

    return np.asarray(output, dtype=float)

def sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))

def train_qadapt_hybrid():
    print("Loading High-Dimensional Leukemia Dataset...")
    df = pd.read_csv("data/raw/golub_leukemia_7129_genes.csv")
    
    # Encode target (Predicting Lung Cancer Tissue vs Other)
    df["Target"] = (df["Tissue"] == "Lung").astype(int) # Other=0, Lung=1
    y = df["Target"].values
    X = df.drop(columns=["Tissue", "Target"])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Applying QADAPT Feature Bottleneck (SelectKBest k=12)...")
    selector = SelectKBest(f_classif, k=12).fit(X_train, y_train)
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"Top 12 Extracted Genes: {selected_features}")
    
    # QADAPT selects the top 4 for the Quantum Circuit
    selector_q = SelectKBest(f_classif, k=4).fit(X_train, y_train)
    quantum_features = X.columns[selector_q.get_support()].tolist()
    
    X_train_cls = X_train[selected_features]
    X_test_cls = X_test[selected_features]
    
    scaler_cls = StandardScaler().fit(X_train_cls)
    
    print("Training Classical SVM Baseline...")
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    svm.fit(scaler_cls.transform(X_train_cls), y_train)
    
    c_prob_train = svm.predict_proba(scaler_cls.transform(X_train_cls))[:, 1]
    c_prob_test = svm.predict_proba(scaler_cls.transform(X_test_cls))[:, 1]
    
    # Pre-scale Quantum Data to Pi angles
    X_train_q = X_train[quantum_features].values
    scaler_q = StandardScaler().fit(X_train_q)
    X_train_q_scaled = scaler_q.transform(X_train_q) * (np.pi / 2) # Angle mapping
    
    # Loss function for Quantum Residual Correction
    def loss_function(params):
        # params: 16 (circuit) + 4 (weights) + 1 (bias) + 1 (alpha) = 22 params
        circuit_params = params[:16]
        weights = params[16:20]
        bias = params[20]
        alpha = params[21]
        
        total_loss = 0
        for i in range(len(y_train)):
            c_logit = logit(c_prob_train[i])
            q_out = quantum_expectations(X_train_q_scaled[i], circuit_params)
            q_logit = np.dot(q_out, weights) + bias
            
            hybrid_logit = (1 - alpha) * c_logit + alpha * q_logit
            hybrid_prob = sigmoid(hybrid_logit)
            
            # Cross Entropy Loss
            y_i = y_train[i]
            loss = - (y_i * np.log(hybrid_prob + 1e-9) + (1 - y_i) * np.log(1 - hybrid_prob + 1e-9))
            total_loss += loss
            
        return total_loss / len(y_train)

    print("Training Quantum Parameter Weights (Residual Correction)...")
    initial_params = np.random.uniform(-0.1, 0.1, 22)
    
    # Using COBYLA for speed in this demo
    res = minimize(loss_function, initial_params, method='COBYLA', options={'maxiter': 50, 'disp': True})
    best_params = res.x
    
    print("\nTraining Complete! Saving QADAPT parameters...")
    np.save("data/genomic_quantum_parameters.npy", best_params)
    
    # Evaluate Hybrid
    X_test_q_scaled = scaler_q.transform(X_test[quantum_features].values) * (np.pi / 2)
    
    y_pred_hybrid = []
    y_prob_hybrid = []
    y_prob_classical = []
    
    circuit_params = best_params[:16]
    weights = best_params[16:20]
    bias = best_params[20]
    alpha = best_params[21]
    
    for i in range(len(y_test)):
        c_logit = logit(c_prob_test[i])
        q_out = quantum_expectations(X_test_q_scaled[i], circuit_params)
        q_logit = np.dot(q_out, weights) + bias
        hybrid_logit = (1 - alpha) * c_logit + alpha * q_logit
        hybrid_prob = sigmoid(hybrid_logit)
        
        y_prob_classical.append(c_prob_test[i])
        y_prob_hybrid.append(hybrid_prob)
        y_pred_hybrid.append(1 if hybrid_prob > 0.5 else 0)
        
    acc = accuracy_score(y_test, y_pred_hybrid)
    print(f"\nFinal Hybrid QADAPT Genomic Accuracy: {acc * 100:.2f}%")
    
    # ---------------------------------------------------------
    # GENERATE UI ARTIFACTS
    # ---------------------------------------------------------
    from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
    
    c_pred = (np.array(y_prob_classical) > 0.5).astype(int)
    
    def get_metrics(y_true, y_p, y_prob):
        tn, fp, fn, tp = confusion_matrix(y_true, y_p).ravel()
        return [
            accuracy_score(y_true, y_p),
            precision_score(y_true, y_p, zero_division=0),
            recall_score(y_true, y_p, zero_division=0), # sensitivity
            tn / (tn + fp) if (tn + fp) > 0 else 0, # specificity
            2 * (precision_score(y_true, y_p, zero_division=0) * recall_score(y_true, y_p, zero_division=0)) / (precision_score(y_true, y_p, zero_division=0) + recall_score(y_true, y_p, zero_division=0) + 1e-9), # f1
            roc_auc_score(y_true, y_prob),
            tn, fp, fn, tp
        ]
        
    cm_metrics = get_metrics(y_test, c_pred, y_prob_classical)
    qm_metrics = get_metrics(y_test, y_pred_hybrid, y_prob_hybrid)
    
    results_df = pd.DataFrame([
        ["Classical_12", 0.5, *cm_metrics],
        ["Residual_Hybrid_V12", 0.5, *qm_metrics]
    ], columns=["model","threshold","accuracy","precision","sensitivity","specificity","f1","auc","tn","fp","fn","tp"])
    
    results_df.to_csv("../QADAPT/12_v12_final_results.csv", index=False)
    
    preds_df = pd.DataFrame({
        "y_true": y_test,
        "prob_Classical_12": y_prob_classical,
        "prob_Residual_Hybrid_V12": y_prob_hybrid
    })
    preds_df.to_csv("../QADAPT/12_v12_predictions.csv", index=False)
    
    print("Exported UI Artifacts to QADAPT folder successfully!")

if __name__ == "__main__":
    train_qadapt_hybrid()
