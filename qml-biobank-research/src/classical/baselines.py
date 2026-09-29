from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def get_logistic_regression():
    model = LogisticRegression(max_iter=1000, random_state=42)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2']
    }
    return GridSearchCV(model, param_grid, cv=5, scoring='average_precision', n_jobs=-1)

def get_random_forest():
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    return GridSearchCV(model, param_grid, cv=5, scoring='average_precision', n_jobs=-1)

def get_classical_svm():
    # probability=True is required to calculate ROC-AUC and PR-AUC
    model = SVC(probability=True, random_state=42)
    param_grid = [
        {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 0.01]},
        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}
    ]
    return GridSearchCV(model, param_grid, cv=5, scoring='average_precision', n_jobs=-1)

def get_xgboost():
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    return GridSearchCV(model, param_grid, cv=5, scoring='average_precision', n_jobs=-1)

def get_all_classical_baselines():
    """
    Returns a dictionary of un-fitted, cross-validated classical baselines ready to be trained.
    """
    return {
        "Logistic_Regression": get_logistic_regression(),
        "Random_Forest": get_random_forest(),
        "Classical_SVM": get_classical_svm(),
        "XGBoost": get_xgboost()
    }

if __name__ == "__main__":
    print("Classical baselines module loaded. Ready to ruthlessly optimize hyperparameters.")
