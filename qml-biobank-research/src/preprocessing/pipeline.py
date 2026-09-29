import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA

def get_preprocessing_pipeline(numeric_cols, categorical_cols, n_pca_components=None, use_minmax=False):
    """
    Constructs a strict scikit-learn Pipeline that absolutely prevents data leakage.
    Transformations (Imputation, Scaling, PCA) are ONLY fitted on the training split.
    
    Args:
        numeric_cols (list): List of numerical feature names.
        categorical_cols (list): List of categorical feature names.
        n_pca_components (int, optional): If set, applies PCA dimensionality reduction. 
                                          CRITICAL for the 'Quantum Bottleneck'.
        use_minmax (bool): If True, uses MinMaxScaler (useful for angle encoding in QML).
                           Otherwise uses StandardScaler.
    Returns:
        sklearn.pipeline.Pipeline
    """
    
    # 1. Numeric Pipeline: Impute missing values with median, then scale.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler() if use_minmax else StandardScaler())
    ])

    # 2. Categorical Pipeline: Impute missing with 'missing' flag, then one-hot encode.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Combine them using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop' # Drop any columns not explicitly specified
    )

    # 4. Optional PCA for extreme dimensionality reduction (The Quantum Bottleneck)
    steps = [('preprocessor', preprocessor)]
    
    if n_pca_components is not None:
        steps.append(('pca', PCA(n_components=n_pca_components)))

    return Pipeline(steps=steps)

def apply_train_test_split_and_preprocess(X, y, numeric_cols, categorical_cols, n_pca_components=None, use_minmax=False, test_size=0.2, random_state=42):
    """
    Utility to securely split data and apply the pipeline, guaranteeing no test leakage.
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    pipeline = get_preprocessing_pipeline(numeric_cols, categorical_cols, n_pca_components, use_minmax)
    
    # Fit ONLY on training data
    X_train_processed = pipeline.fit_transform(X_train)
    
    # Transform test data using the patterns learned from training data
    X_test_processed = pipeline.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    print("Preprocessing module loaded. Ready to prevent data leakage.")
