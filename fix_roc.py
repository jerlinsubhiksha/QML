import os

files = ['merged_hybrid_qml.py', 'qml_breast_cancer.py', 'qml_custom.py']

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Imports
    if 'roc_curve' not in content:
        content = content.replace('roc_auc_score,', 'roc_auc_score,\n    roc_curve,')
        
    # 2. calculate_metrics function body
    old_calc = '''    try: auc = roc_auc_score(y_true, y_score)
    except: auc = 0.0'''
    new_calc = '''    try: 
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except: 
        auc = 0.0
        fpr, tpr = [0, 1], [0, 1]'''
    content = content.replace(old_calc, new_calc)
    
    old_calc2 = '''    try: auc = roc_auc_score(y_true, y_score)
    except Exception: auc = 0.0'''
    content = content.replace(old_calc2, new_calc)
    
    # Add fpr and tpr to the returned dict
    if '"fpr":' not in content:
        content = content.replace('"auc": auc,', '"auc": auc,\n        "fpr": ",".join(map(str, fpr)),\n        "tpr": ",".join(map(str, tpr)),')
        
    # 3. Append logic
    content = content.replace(
        '"ROC-AUC": res["auc"].mean(),\n            "Runtime": res["runtime"].mean(),',
        '"ROC-AUC": res["auc"].mean(),\n            "FPR": res["fpr"].iloc[-1],\n            "TPR": res["tpr"].iloc[-1],\n            "Runtime": res["runtime"].mean(),'
    )
    content = content.replace(
        '"ROC-AUC": q_res["auc"].mean(),\n        "Runtime": q_res["runtime"].mean(),',
        '"ROC-AUC": q_res["auc"].mean(),\n        "FPR": q_res["fpr"].iloc[-1],\n        "TPR": q_res["tpr"].iloc[-1],\n        "Runtime": q_res["runtime"].mean(),'
    )
    content = content.replace(
        '"ROC-AUC": q_res["auc"].mean(),\n            "Runtime": q_res["runtime"].mean(),',
        '"ROC-AUC": q_res["auc"].mean(),\n            "FPR": q_res["fpr"].iloc[-1],\n            "TPR": q_res["tpr"].iloc[-1],\n            "Runtime": q_res["runtime"].mean(),'
    )
    
    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)
print('Done!')
