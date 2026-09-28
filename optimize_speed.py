import os
files = ['merged_hybrid_qml.py', 'qml_breast_cancer.py', 'qml_custom.py']

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Reduce N_SAMPLES for Fever dataset
    content = content.replace('N_SAMPLES = 120', 'N_SAMPLES = 60')
    
    # 2. Reduce folds to 2
    content = content.replace('N_FOLDS = 3', 'N_FOLDS = 2')
    
    # 3. Reduce Breast Cancer downsample from 50 to 40
    content = content.replace('size=50', 'size=40')
    
    # 4. Reduce Custom Dataset downsample from 100 to 50
    content = content.replace('size=100', 'size=50')
    content = content.replace('len(X) > 100:', 'len(X) > 50:')
    
    # 5. Speed up QSVM
    content = content.replace('reps=2', 'reps=1')
    content = content.replace('entanglement="full"', 'entanglement="linear"')

    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)
print('Optimizations applied.')
