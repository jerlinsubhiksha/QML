import os
import urllib.request
import tarfile
import pandas as pd

def download_tcga_rnaseq():
    """
    Instantly downloads the TCGA Pan-Cancer RNA-Seq dataset from the UCI Repository.
    This is a massive, real-world genomics dataset (801 patients, 20,531 genes) 
    that is perfectly suited for testing QML dimensionality reduction and entanglement.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801data.tar.gz"
    data_dir = os.path.join("data", "raw", "TCGA_RNASeq")
    os.makedirs(data_dir, exist_ok=True)
    
    tar_path = os.path.join(data_dir, "TCGA-PANCAN.tar.gz")
    
    print("Downloading TCGA RNA-Seq Genomic Dataset (this may take a minute)...")
    urllib.request.urlretrieve(url, tar_path)
    
    print("Extracting genomic data...")
    with tarfile.open(tar_path, "r:gz") as tar:
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=data_dir)
    
    # The tar file extracts to a folder called 'TCGA-PANCAN-HiSeq-801data'
    extracted_folder = os.path.join(data_dir, "TCGA-PANCAN-HiSeq-801data")
    
    # Load and merge data + labels
    print("Formatting into a single matrix...")
    df_data = pd.read_csv(os.path.join(extracted_folder, "data.csv"))
    df_labels = pd.read_csv(os.path.join(extracted_folder, "labels.csv"))
    
    # Merge on the first column (patient ID)
    df_final = pd.merge(df_labels, df_data, on="Unnamed: 0")
    
    # Rename for clarity
    df_final.rename(columns={"Unnamed: 0": "Patient_ID", "Class": "Cancer_Type"}, inplace=True)
    
    final_csv_path = os.path.join("data", "raw", "tcga_pan_cancer_rna_seq.csv")
    df_final.to_csv(final_csv_path, index=False)
    
    print(f"\nSUCCESS! High-dimensional genomic dataset saved to: {final_csv_path}")
    print(f"Shape: {df_final.shape} (Patients x Genes)")
    print("Cancer Subtypes:", df_final['Cancer_Type'].unique())
    
    # Cleanup
    os.remove(tar_path)

if __name__ == "__main__":
    download_tcga_rnaseq()
