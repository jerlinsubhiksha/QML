import time
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

warnings.filterwarnings("ignore")

def generate_quantum_advantage_data(num_samples=300):
    """
    Generates a synthetic dataset mathematically designed to mimic quantum interference.
    Classical algorithms (like RBF SVM) struggle to find the decision boundary because 
    it oscillates rapidly in a periodic space. The Quantum SVM, using angle encoding, 
    maps this natively to qubit phases and solves it easily.
    """
    rng = np.random.default_rng(42)
    
    # 2 Features: angles between 0 and 2π
    X = rng.uniform(0, 2 * np.pi, (num_samples, 2))
    
    # Complex non-linear labels based on phase interference
    y = np.zeros(num_samples)
    for i in range(num_samples):
        x1, x2 = X[i]
        # The decision boundary is a highly complex trigonometric function
        interference = np.cos(x1) * np.cos(x2) - np.sin(x1) * np.sin(x2) + np.sin(x1 * x2)
        y[i] = 1 if interference > 0 else 0
        
    return X, y

def main():
    print("==================================================")
    print("   THE 'QUANTUM ADVANTAGE' DATASET DEMONSTRATION")
    print("==================================================")
    print("Generating a dataset based on quantum interference...\n")
    
    X, y = generate_quantum_advantage_data(300)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ---------------------------------------------------------
    # 1. CLASSICAL SVM (RBF Kernel)
    # ---------------------------------------------------------
    print("--- Training CLASSICAL SVM (RBF Kernel) ---")
    start = time.perf_counter()
    
    classical_svc = SVC(kernel='rbf')
    classical_svc.fit(X_train, y_train)
    c_preds = classical_svc.predict(X_test)
    c_acc = accuracy_score(y_test, c_preds)
    
    print(f"Classical SVM Accuracy : {c_acc * 100:.2f}%")
    print(f"Time Taken             : {time.perf_counter() - start:.2f}s\n")

    # ---------------------------------------------------------
    # 2. QUANTUM SVM (Fidelity Quantum Kernel)
    # ---------------------------------------------------------
    print("--- Training QUANTUM SVM (ZZFeatureMap) ---")
    start = time.perf_counter()
    
    # 2 Qubits, deep entanglement. This feature map natively understands 
    # the periodic sine/cosine relationships in the dataset.
    feature_map = ZZFeatureMap(feature_dimension=2, reps=3, entanglement='full')
    qkernel = FidelityQuantumKernel(feature_map=feature_map)
    
    qsvm = QSVC(quantum_kernel=qkernel)
    qsvm.fit(X_train, y_train)
    q_preds = qsvm.predict(X_test)
    q_acc = accuracy_score(y_test, q_preds)
    
    print(f"Quantum SVM Accuracy   : {q_acc * 100:.2f}%")
    print(f"Time Taken             : {time.perf_counter() - start:.2f}s\n")
    
    print("==================================================")
    if q_acc > c_acc:
        print("🏆 RESULT: QUANTUM ML FINALLY BEAT CLASSICAL ML! 🏆")
    else:
        print("RESULT: Classical ML held on.")
    print("==================================================")

if __name__ == "__main__":
    main()
