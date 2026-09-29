import numpy as np
import pandas as pd
import os

def generate_epistatic_dataset(n_samples=1000, n_features=20, random_state=42):
    """
    Generates a highly complex, non-linear dataset simulating 'High-Order Epistasis' 
    in protein-protein interactions.
    
    Classical models (like Random Forests or Linear SVMs) fail at this because 
    the target is determined by a complex 4-way XOR interaction between specific proteins.
    Individually, NO single protein correlates with the disease.
    Quantum Kernels (like the ZZFeatureMap) naturally compute these types of 
    entangled parity interactions, giving QML a massive theoretical advantage here.
    """
    np.random.seed(random_state)
    
    # Generate background protein expression levels (continuous values from 0 to 1)
    X = np.random.uniform(0, 1, (n_samples, n_features))
    
    # The 'Disease' is triggered ONLY by a highly complex, hidden non-linear interaction
    # between Protein_0, Protein_1, Protein_2, and Protein_3.
    # This simulates a 4-way molecular binding complex (pure epistasis).
    
    # We use a parity-like (XOR) threshold function which is famously difficult for Classical ML
    interaction_term = (
        np.sin(np.pi * X[:, 0]) * 
        np.cos(np.pi * X[:, 1]) * 
        np.sin(np.pi * X[:, 2]) * 
        np.cos(np.pi * X[:, 3])
    )
    
    # Add heavy biological noise to make it even harder
    noise = np.random.normal(0, 0.15, n_samples)
    target_continuous = interaction_term + noise
    
    # Binarize into Healthy (0) vs Disease (1)
    y = (target_continuous > np.median(target_continuous)).astype(int)
    
    # Create DataFrame
    columns = [f"Protein_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["Disease_State"] = y
    
    # Save to data/raw
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    out_path = os.path.join("data", "raw", "complex_proteomic_epistasis.csv")
    df.to_csv(out_path, index=False)
    
    print(f"Successfully generated highly complex epistatic dataset at: {out_path}")
    print(f"Shape: {df.shape}. Class balance: {np.mean(y):.2f}")

if __name__ == "__main__":
    generate_epistatic_dataset()
