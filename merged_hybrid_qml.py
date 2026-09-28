import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from qiskit.circuit.library import PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION
# ================================================================
RANDOM_STATE = 42
N_SAMPLES = 120
N_FOLDS = 3 # Reduced to 3 to speed up quantum simulation
N_QUBITS = 4 # We will use PCA to reduce the 6 fever features to 4 qubits

# ================================================================
# 1. GENERATE A ROUGH BIOMEDICAL DATASET
# ================================================================
def generate_dataset():
    print("\n" + "=" * 70)
    print("GENERATING FEVER / ACUTE INFLAMMATION DATA")
    print("=" * 70)

    rng = np.random.default_rng(RANDOM_STATE)
    data = []

    for _ in range(N_SAMPLES):
        temperature = np.clip(rng.normal(loc=38.0, scale=1.2), 35.5, 41.5)
        nausea = rng.binomial(1, 0.35)
        lumbar_pain = rng.binomial(1, 0.35)
        urine_pushing = rng.binomial(1, 0.40)
        micturition_pain = rng.binomial(1, 0.40)
        burning_urethra = rng.binomial(1, 0.40)

        risk = 0
        if temperature >= 38.5: risk += 2
        elif temperature >= 37.8: risk += 1
        
        risk += nausea + (lumbar_pain * 2) + urine_pushing + micturition_pain + burning_urethra + rng.normal(0, 1.0)
        inflammation = 1 if risk >= 4 else 0

        data.append([temperature, nausea, lumbar_pain, urine_pushing, micturition_pain, burning_urethra, inflammation])

    columns = ["temperature", "nausea", "lumbar_pain", "urine_pushing", "micturition_pain", "burning_urethra", "inflammation"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("rough_fever_dataset.csv", index=False)
    
    print("Dataset created successfully. Shape:", df.shape)
    return df

# ================================================================
# 2. PREPARE DATA
# ================================================================
def prepare_data(df):
    feature_columns = ["temperature", "nausea", "lumbar_pain", "urine_pushing", "micturition_pain", "burning_urethra"]
    X = df[feature_columns].to_numpy(dtype=float)
    y = df["inflammation"].to_numpy(dtype=int)
    return X, y

def calculate_metrics(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try: auc = roc_auc_score(y_true, y_score)
    except: auc = 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

# ================================================================
# 3. CLASSICAL MACHINE LEARNING
# ================================================================
def run_classical_svm(X, y):
    print("\n" + "=" * 70)
    print("CLASSICAL MACHINE LEARNING (SVM)")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    total_start = time.perf_counter()

    for fold, (train_index, test_index) in enumerate(cv.split(X, y), start=1):
        fold_start = time.perf_counter()
        
        # We integrate the PCA from our custom code to make it mathematically fair for QML
        pca = PCA(n_components=N_QUBITS)
        X_train_sel = pca.fit_transform(X[train_index])
        X_test_sel = pca.transform(X[test_index])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)

        model = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
        model.fit(X_train_scaled, y[train_index])
        
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]

        metrics = calculate_metrics(y[test_index], predictions, probabilities)
        fold_time = time.perf_counter() - fold_start
        metrics["runtime"] = fold_time
        fold_results.append(metrics)
        
        print(f"Classical Fold {fold}/{N_FOLDS} completed in {fold_time:.2f}s | Acc: {metrics['accuracy']:.4f}")

    results = pd.DataFrame(fold_results)
    print(f"\nClassical SVM Accuracy: {results['accuracy'].mean():.4f} +/- {results['accuracy'].std():.4f}")
    return results

# ================================================================
# 4. QUANTUM MACHINE LEARNING
# ================================================================
def run_qsvm(X, y):
    print("\n" + "=" * 70)
    print("QUANTUM MACHINE LEARNING (QSVM)")
    print("=" * 70)

    # --- MERGED UPGRADE: PAULI FEATURE MAP WITH FULL ENTANGLEMENT ---
    feature_map = PauliFeatureMap(feature_dimension=N_QUBITS, reps=2, paulis=['Z', 'YY'], entanglement="full")
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    total_start = time.perf_counter()

    for fold, (train_index, test_index) in enumerate(cv.split(X, y), start=1):
        fold_start = time.perf_counter()

        # --- MERGED UPGRADE: PCA ---
        pca = PCA(n_components=N_QUBITS)
        X_train_sel = pca.fit_transform(X[train_index])
        X_test_sel = pca.transform(X[test_index])

        # --- MERGED UPGRADE: MINMAX SCALER FOR ANGLES (0 to Pi) ---
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
        
        print(f"Quantum Fold {fold}/{N_FOLDS} completed in {fold_time:.2f}s | Acc: {metrics['accuracy']:.4f}")

    results = pd.DataFrame(fold_results)
    print(f"\nQuantum SVM Accuracy: {results['accuracy'].mean():.4f} +/- {results['accuracy'].std():.4f}")
    return results

def main():
    df = generate_dataset()
    X, y = prepare_data(df)

    c_results = run_classical_svm(X, y)
    q_results = run_qsvm(X, y)

    # Combine into the format our UI expects
    summary = []
    
    summary.append({
        "Dataset": "Fever/Acute Inflammation",
        "Model": "Classical SVM",
        "Accuracy": c_results["accuracy"].mean(),
        "Sensitivity": c_results["sensitivity"].mean(),
        "Precision": c_results["precision"].mean(),
        "Runtime": c_results["runtime"].mean(),
        "TN": c_results["tn"].sum(),
        "FP": c_results["fp"].sum(),
        "FN": c_results["fn"].sum(),
        "TP": c_results["tp"].sum()
    })
    
    summary.append({
        "Dataset": "Fever/Acute Inflammation",
        "Model": "Quantum SVM",
        "Accuracy": q_results["accuracy"].mean(),
        "Sensitivity": q_results["sensitivity"].mean(),
        "Precision": q_results["precision"].mean(),
        "Runtime": q_results["runtime"].mean(),
        "TN": q_results["tn"].sum(),
        "FP": q_results["fp"].sum(),
        "FN": q_results["fn"].sum(),
        "TP": q_results["tp"].sum()
    })
    
    summary_df = pd.DataFrame(summary)
    import os
    file_exists = os.path.isfile("master_qml_results.csv")
    summary_df.to_csv("master_qml_results.csv", mode='a', header=not file_exists, index=False)
    print("\nResults successfully appended to master_qml_results.csv for the UI to read!")

if __name__ == "__main__":
    main()
