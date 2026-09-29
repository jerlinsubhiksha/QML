import os
import pandas as pd
from sklearn.datasets import fetch_openml

def download_real_leukemia_genomics():
    """
    Downloads the famous Golub (1999) Leukemia Microarray dataset.
    This dataset contains 7,129 gene expression features for 72 patients.
    It classifies Acute Lymphoblastic Leukemia (ALL) vs Acute Myeloid Leukemia (AML).
    Because it has 7000+ dimensions but only 72 samples, it is incredibly complex 
    and perfect for testing Quantum PCA and ZZFeatureMap entanglement.
    """
    print("Fetching the Golub Leukemia Microarray Dataset from OpenML...")
    print("This contains 7,129 real gene expression levels per patient.")
    
    # fetch_openml ID 1130 is the Leukemia dataset
    leukemia = fetch_openml(data_id=1130, as_frame=True, parser='auto')
    
    df = leukemia.frame
    # The target column is named 'class'
    
    # Ensure directory exists
    data_dir = os.path.join("data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    
    out_path = os.path.join(data_dir, "golub_leukemia_7129_genes.csv")
    df.to_csv(out_path, index=False)
    
    print(f"\nSUCCESS! Real Genomic dataset saved to: {out_path}")
    print(f"Shape: {df.shape[0]} Patients x {df.shape[1] - 1} Genes")
    print("Classes:", df['class'].unique())

if __name__ == "__main__":
    download_real_leukemia_genomics()
