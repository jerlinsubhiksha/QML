import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION
# ================================================================
RANDOM_STATE = 42
N_FOLDS = 3 # Reduced to 3 to speed up the quantum simulation, QSVM can be slow
N_QUBITS = 4 # Reduced to 4 to make the simulation blazing fast for the UI

# ================================================================
# 1. LOAD DATASETS
# ================================================================

def get_datasets():
    print("=" * 70)
    print("LOADING BREAST CANCER DATASET & GENERATING NOISY VERSION")
    print("=" * 70)
    
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # --- CRITICAL FIX FOR QUANTUM SIMULATION ---
    # Simulating a quantum kernel for 569 samples requires over 320,000 quantum circuit calculations.
    # This takes hours on a classical CPU. We downsample to 100 patients to make it run in a few minutes.
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=50, replace=False)
    X = X[idx]
    y = y[idx]
    
    # Dataset 1: Clean Breast Cancer Data
    # Dataset 2: Heavy Noise added to the features
    
    # We add significant gaussian noise to the normalized features to make it "heavy noise"
    # To do this fairly, we calculate the standard deviation of each feature and add noise proportional to it.
    X_noisy = X.copy()
    stds = np.std(X, axis=0)
    # Add noise equal to 1.5 * standard deviation of each feature
    noise = rng.normal(loc=0, scale=1.5 * stds, size=X.shape)
    X_noisy = X_noisy + noise
    
    datasets = {
        "Clean Breast Cancer Dataset": (X, y),
        "Heavy Noise Medical Dataset": (X_noisy, y)
    }
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features.")
    print("Created 'Clean' and 'Heavy Noise' variations.\n")
    return datasets

# ================================================================
# 2. METRIC FUNCTION
# ================================================================

def calculate_metrics(y_true, y_pred, y_score):
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn = cm[0, 0]
        fp = cm[0, 1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": auc
    }

# ================================================================
# 3. CLASSICAL MACHINE LEARNING
# ================================================================

def run_classical_svm(X, y, dataset_name):
    print(f"\n--- CLASSICAL SVM ON: {dataset_name} ---")
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    
    total_start = time.perf_counter()
    for fold, (train_index, test_index) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fold_start = time.perf_counter()

        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=N_QUBITS)
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)

        # SVM
        model = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]

        metrics = calculate_metrics(y_test, predictions, probabilities)
        fold_time = time.perf_counter() - fold_start
        metrics["runtime"] = fold_time
        fold_results.append(metrics)
        
        print(f"  -> Fold {fold}/{N_FOLDS} completed in {fold_time:.2f}s | Acc: {metrics['accuracy']:.4f} | Sens: {metrics['sensitivity']:.4f} | Prec: {metrics['precision']:.4f}")
        
    total_time = time.perf_counter() - total_start
    results = pd.DataFrame(fold_results)
    print(f"Classical SVM Accuracy: {results['accuracy'].mean():.4f} +/- {results['accuracy'].std():.4f} in {total_time:.2f}s")
    return results

# ================================================================
# 4. QUANTUM MACHINE LEARNING
# ================================================================

def run_qsvm(X, y, dataset_name):
    print(f"\n--- QUANTUM SVM ON: {dataset_name} ---")
    
    # --- IMPROVEMENT 5: PAULI FEATURE MAP ---
    # We replaced the standard ZZFeatureMap with a PauliFeatureMap using 'Z' and 'YY' interactions.
    # This creates a more complex, non-linear mapping into the quantum state space that is harder
    # for classical computers to replicate.
    feature_map = PauliFeatureMap(feature_dimension=N_QUBITS, reps=1, paulis=['Z', 'YY'], entanglement="linear")
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    
    total_start = time.perf_counter()
    for fold, (train_index, test_index) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fold_start = time.perf_counter()

        # --- IMPROVEMENT 2: USE PCA INSTEAD OF SELECTKBEST ---
        # Instead of just picking the 4 "best" features, PCA compresses the information
        # of ALL 30 features down into 4 components, preserving much more medical data.
        from sklearn.decomposition import PCA
        selector = PCA(n_components=N_QUBITS)
        X_train_sel = selector.fit_transform(X_train)
        X_test_sel = selector.transform(X_test)

        # Scaling
        from sklearn.preprocessing import MinMaxScaler
        # --- IMPROVEMENT 3: OPTIMAL SCALING FOR QUANTUM ANGLES ---
        # Quantum gates take angles between 0 and 2π. MinMaxScaler is better than StandardScaler here.
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train_quantum = scaler.fit_transform(X_train_sel)
        X_test_quantum = scaler.transform(X_test_sel)

        # QSVM
        qsvm = QSVC(quantum_kernel=quantum_kernel)
        qsvm.fit(X_train_quantum, y_train)
        
        predictions = qsvm.predict(X_test_quantum)
        scores = predictions.astype(float) # QSVC binary predictions as score

        metrics = calculate_metrics(y_test, predictions, scores)
        fold_time = time.perf_counter() - fold_start
        metrics["runtime"] = fold_time
        fold_results.append(metrics)
        
        print(f"  -> Fold {fold}/{N_FOLDS} completed in {fold_time:.2f}s | Acc: {metrics['accuracy']:.4f} | Sens: {metrics['sensitivity']:.4f} | Prec: {metrics['precision']:.4f}")
        
    total_time = time.perf_counter() - total_start
    results = pd.DataFrame(fold_results)
    print(f"Quantum SVM Accuracy: {results['accuracy'].mean():.4f} +/- {results['accuracy'].std():.4f} in {total_time:.2f}s")
    return results

# ================================================================
# 5. MAIN
# ================================================================

def main():
    datasets = get_datasets()
    
    summary = []
    
    for name, (X, y) in datasets.items():
        print(f"\n{'='*70}\nEVALUATING: {name}\n{'='*70}")
        
        c_results = run_classical_svm(X, y, name)
        q_results = run_qsvm(X, y, name)
        
        summary.append({
            "Dataset": name,
            "Model": "Classical SVM",
            "Accuracy": c_results["accuracy"].mean(),
            "Sensitivity": c_results["sensitivity"].mean(),
            "Precision": c_results["precision"].mean()
        })
        
        summary.append({
            "Dataset": name,
            "Model": "Quantum SVM",
            "Accuracy": q_results["accuracy"].mean(),
            "Sensitivity": q_results["sensitivity"].mean(),
            "Precision": q_results["precision"].mean()
        })
        
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    import os
    file_exists = os.path.isfile("master_qml_results.csv")
    summary_df.to_csv("master_qml_results.csv", mode='a', header=not file_exists, index=False)
    print("\nResults appended to master_qml_results.csv")

if __name__ == "__main__":
    main()
