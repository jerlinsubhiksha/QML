import argparse
import time
import warnings
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from qiskit.circuit.library import PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_FOLDS = 2 
N_QUBITS = 4 

def calculate_metrics(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        specificity = 0.0

    try: 
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except: 
        auc = 0.0
        fpr, tpr = [0, 1], [0, 1]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0, average='weighted'),
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0, average='weighted'),
        "f1": f1_score(y_true, y_pred, zero_division=0, average='weighted'),
        "auc": auc,
        "fpr": ",".join(map(str, fpr)),
        "tpr": ",".join(map(str, tpr)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

def run_classical_model(model, name, X, y):
    print(f"\n--- {name.upper()} ---")
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []

    for fold, (train_index, test_index) in enumerate(cv.split(X, y), start=1):
        fold_start = time.perf_counter()
        
        pca = PCA(n_components=min(N_QUBITS, X.shape[1]))
        X_train_sel = pca.fit_transform(X[train_index])
        X_test_sel = pca.transform(X[test_index])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)

        model.fit(X_train_scaled, y[train_index])
        
        predictions = model.predict(X_test_scaled)
        try: probabilities = model.predict_proba(X_test_scaled)[:, 1]
        except: probabilities = np.zeros(len(predictions))

        metrics = calculate_metrics(y[test_index], predictions, probabilities)
        fold_time = time.perf_counter() - fold_start
        metrics["runtime"] = fold_time
        fold_results.append(metrics)
        print(f"Fold {fold} | Acc: {metrics['accuracy']:.4f} | Time: {fold_time:.2f}s")

    return pd.DataFrame(fold_results)

def run_qsvm(X, y):
    print("\n--- QUANTUM SVM ---")
    actual_qubits = min(N_QUBITS, X.shape[1])
    feature_map = PauliFeatureMap(feature_dimension=actual_qubits, reps=1, paulis=['Z', 'YY'], entanglement="linear")
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []

    for fold, (train_index, test_index) in enumerate(cv.split(X, y), start=1):
        fold_start = time.perf_counter()

        pca = PCA(n_components=actual_qubits)
        X_train_sel = pca.fit_transform(X[train_index])
        X_test_sel = pca.transform(X[test_index])

        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train_quantum = scaler.fit_transform(X_train_sel)
        X_test_quantum = scaler.transform(X_test_sel)

        qsvm = QSVC(quantum_kernel=quantum_kernel)
        qsvm.fit(X_train_quantum, y[train_index])
        
        predictions = qsvm.predict(X_test_quantum)
        scores = predictions.astype(float)

        metrics = calculate_metrics(y[test_index], predictions, scores)
        fold_time = time.perf_counter() - fold_start
        metrics["runtime"] = fold_time
        fold_results.append(metrics)
        print(f"Fold {fold} | Acc: {metrics['accuracy']:.4f} | Time: {fold_time:.2f}s")

    return pd.DataFrame(fold_results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--name", default="Custom Dataset", help="Dataset name for UI")
    args = parser.parse_args()

    print(f"Loading custom dataset: {args.data}")
    df = pd.read_csv(args.data)
    df = df.dropna()
    
    if args.target not in df.columns:
        print(f"Error: Target column '{args.target}' not found in dataset.")
        return

    if df[args.target].dtype == 'object':
        df[args.target] = df[args.target].astype('category').cat.codes

    y = df[args.target].to_numpy(dtype=int)
    X_df = df.drop(columns=[args.target])
    X_df = X_df.select_dtypes(include=[np.number])
    X = X_df.to_numpy(dtype=float)

    if len(X) > 50:
        print(f"Dataset has {len(X)} rows. Downsampling to 100 for fast Quantum simulation...")
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X), size=50, replace=False)
        X = X[idx]
        y = y[idx]

    models = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Classical SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    }

    summary = []
    
    for name, model in models.items():
        res = run_classical_model(model, name, X, y)
        summary.append({
            "Dataset": args.name,
            "Model": name,
            "Accuracy": res["accuracy"].mean(),
            "Sensitivity": res["sensitivity"].mean(),
            "Specificity": res["specificity"].mean(),
            "Precision": res["precision"].mean(),
            "F1-score": res["f1"].mean(),
            "ROC-AUC": res["auc"].mean(),
            "FPR": res["fpr"].iloc[-1],
            "TPR": res["tpr"].iloc[-1],
            "Runtime": res["runtime"].mean(),
            "TN": res["tn"].sum(),
            "FP": res["fp"].sum(),
            "FN": res["fn"].sum(),
            "TP": res["tp"].sum()
        })

    q_res = run_qsvm(X, y)
    summary.append({
        "Dataset": args.name,
        "Model": "Quantum SVM",
        "Accuracy": q_res["accuracy"].mean(),
        "Sensitivity": q_res["sensitivity"].mean(),
        "Specificity": q_res["specificity"].mean(),
        "Precision": q_res["precision"].mean(),
        "F1-score": q_res["f1"].mean(),
        "ROC-AUC": q_res["auc"].mean(),
        "FPR": q_res["fpr"].iloc[-1],
        "TPR": q_res["tpr"].iloc[-1],
        "Runtime": q_res["runtime"].mean(),
        "TN": q_res["tn"].sum(),
        "FP": q_res["fp"].sum(),
        "FN": q_res["fn"].sum(),
        "TP": q_res["tp"].sum()
    })
    
    summary_df = pd.DataFrame(summary)
    file_exists = os.path.isfile("master_qml_results.csv")
    summary_df.to_csv("master_qml_results.csv", mode='a', header=not file_exists, index=False)
    print("\nResults successfully saved!")

if __name__ == "__main__":
    main()
