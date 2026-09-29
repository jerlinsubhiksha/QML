"""
V12 HYBRID QML — REAL INFERENCE ENGINE
======================================

This is the reusable inference core for the final platform.

It reconstructs the exact V12 development-time preprocessing/model setup
from the original dataset and the saved V12 quantum parameters.

Pipeline:
    Patient data
        ↓
    V12 12-feature selection
        ↓
    Classical SVM
        ↓
    Classical probability
        +
    V12 quantum residual correction
        ↓
    Hybrid probability
        ↓
    Prediction + explanation

Research/decision-support prototype only.
NOT a clinical diagnostic system.
"""

import os
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp


# ============================================================
# CONFIGURATION — MATCHES V12
# ============================================================

DATA_PATH = os.path.expanduser("~/Downloads/alzheimers_disease_data.csv")
PARAMETER_PATH = os.path.expanduser("~/Downloads/12_v12_quantum_parameters.npy")
FEATURE_PATH = os.path.expanduser("~/Downloads/12_v12_selected_features.csv")
RESULT_PATH = os.path.expanduser("~/Downloads/12_v12_final_results.csv")

RANDOM_STATE = 42
FINAL_TEST_SIZE = 0.20

N_QUBITS = 4
N_LAYERS = 2

CLASSICAL_THRESHOLD = 0.44
HYBRID_THRESHOLD = 0.45


# ============================================================
# BASIC HELPERS
# ============================================================

def sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


# ============================================================
# V12 QUANTUM CIRCUIT
# ============================================================

def quantum_expectations(x, qparams):
    """
    Exact quantum circuit structure used by V12.
    """

    qc = QuantumCircuit(N_QUBITS)

    # Patient-data encoding
    for q in range(N_QUBITS):
        qc.ry(float(x[q]), q)
        qc.rz(float(0.5 * x[q]), q)

    idx = 0

    # Two trainable layers
    for _ in range(N_LAYERS):

        for q in range(N_QUBITS):

            qc.ry(float(qparams[idx]), q)
            idx += 1

            qc.rz(float(qparams[idx]), q)
            idx += 1

        # Ring entanglement
        for q in range(N_QUBITS - 1):
            qc.cx(q, q + 1)

        qc.cx(N_QUBITS - 1, 0)

        # Data re-uploading
        for q in range(N_QUBITS):
            qc.ry(float(0.5 * x[q]), q)

    state = Statevector.from_instruction(qc)

    output = []

    for q in range(N_QUBITS):

        pauli = ["I"] * N_QUBITS
        pauli[N_QUBITS - 1 - q] = "Z"

        observable = SparsePauliOp.from_list([
            ("".join(pauli), 1.0)
        ])

        value = np.real(
            state.expectation_value(observable)
        )

        output.append(float(value))

    return np.asarray(output, dtype=float)


# ============================================================
# V12 ENGINE
# ============================================================

class V12HybridPredictor:

    def __init__(
        self,
        dataset_path=DATA_PATH,
        parameter_path=PARAMETER_PATH,
        feature_path=FEATURE_PATH,
    ):

        self.dataset_path = dataset_path
        self.parameter_path = parameter_path
        self.feature_path = feature_path

        self.quantum_parameters = None

        self.selected_features = None
        self.quantum_features = None

        self.classical_scaler = None
        self.classical_model = None

        self.quantum_scaler = None

        self.quantum_weights = None
        self.quantum_bias = None
        self.quantum_alpha = None

        self.ready = False

    # ========================================================
    # LOAD V12 ARTIFACTS
    # ========================================================

    def load_artifacts(self):

        required = [
            self.dataset_path,
            self.parameter_path,
            self.feature_path,
        ]

        for path in required:

            if not os.path.exists(path):

                raise FileNotFoundError(
                    f"Required V12 file not found:\n{path}"
                )

        # Quantum parameter vector
        theta = np.load(self.parameter_path)

        if theta.shape != (22,):

            raise ValueError(
                f"Expected 22 V12 quantum parameters, "
                f"found shape {theta.shape}"
            )

        self.quantum_parameters = theta

        # Selected features from V12
        feature_df = pd.read_csv(self.feature_path)

        self.selected_features = (
            feature_df["feature"].tolist()
        )

        self.quantum_features = feature_df.loc[
            feature_df["is_quantum_feature"] == True,
            "feature"
        ].tolist()

        if len(self.selected_features) != 12:
            raise ValueError(
                "V12 artifact does not contain exactly 12 selected features."
            )

        if len(self.quantum_features) != 4:
            raise ValueError(
                "V12 artifact does not contain exactly 4 quantum features."
            )

        # First 8 parameters = quantum circuit
        # Next 4 = quantum weights
        # Next = bias
        # Last = alpha
        n_q_params = N_LAYERS * N_QUBITS * 2

        self.quantum_weights = (
            theta[n_q_params:n_q_params + N_QUBITS]
        )

        self.quantum_bias = float(theta[-2])
        self.quantum_alpha = float(theta[-1])

    # ========================================================
    # RECONSTRUCT FINAL V12 CLASSICAL PIPELINE
    # ========================================================

    def reconstruct_model(self):

        df = pd.read_csv(self.dataset_path)

        y = df["Diagnosis"].astype(int).to_numpy()

        X = df.drop(
            columns=[
                "Diagnosis",
                "PatientID",
                "DoctorInCharge"
            ]
        )

        # EXACT V12 final development split
        X_dev, X_test, y_dev, y_test = train_test_split(
            X,
            y,
            test_size=FINAL_TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        # Verify the saved features against deterministic V12
        selector = SelectKBest(
            f_classif,
            k=12
        ).fit(X_dev, y_dev)

        reconstructed = list(
            X_dev.columns[
                selector.get_support()
            ]
        )

        if reconstructed != self.selected_features:

            raise RuntimeError(
                "Saved V12 feature list does not match "
                "the deterministic V12 reconstruction.\n"
                f"Saved: {self.selected_features}\n"
                f"Reconstructed: {reconstructed}"
            )

        # Final V12 classical model
        self.classical_scaler = StandardScaler().fit(
            X_dev[self.selected_features]
        )

        self.classical_model = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

        self.classical_model.fit(
            self.classical_scaler.transform(
                X_dev[self.selected_features]
            ),
            y_dev,
        )

        # Final V12 quantum scaler
        self.quantum_scaler = StandardScaler().fit(
            X_dev[self.quantum_features]
        )

        self.ready = True

        print("V12 model reconstructed successfully.")

        print(
            f"Classical features: {len(self.selected_features)}"
        )

        print(
            f"Quantum features: {len(self.quantum_features)}"
        )

    # ========================================================
    # START ENGINE
    # ========================================================

    def load(self):

        self.load_artifacts()
        self.reconstruct_model()

        return self

    # ========================================================
    # VALIDATE PATIENT INPUT
    # ========================================================

    def validate_patient(self, patient):

        missing = [
            feature
            for feature in self.selected_features
            if feature not in patient
        ]

        if missing:

            raise ValueError(
                "Missing required patient fields:\n"
                + "\n".join(
                    f"  - {x}" for x in missing
                )
            )

        for feature in self.selected_features:

            value = patient[feature]

            if value is None:

                raise ValueError(
                    f"{feature} cannot be empty."
                )

            try:
                float(value)
            except (TypeError, ValueError):

                raise ValueError(
                    f"{feature} must be numeric."
                )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, patient):

        if not self.ready:
            raise RuntimeError(
                "V12 engine is not loaded. "
                "Call .load() first."
            )

        self.validate_patient(patient)

        patient_df = pd.DataFrame(
            [patient]
        )

        # ----------------------------------------------------
        # Classical probability
        # ----------------------------------------------------

        classical_scaled = self.classical_scaler.transform(
            patient_df[self.selected_features]
        )

        classical_probability = float(
            self.classical_model.predict_proba(
                classical_scaled
            )[0, 1]
        )

        # ----------------------------------------------------
        # Quantum branch
        # ----------------------------------------------------

        quantum_z = self.quantum_scaler.transform(
            patient_df[self.quantum_features]
        )[0]

        quantum_input = (
            np.tanh(quantum_z)
            * (np.pi / 2.0)
        )

        quantum_output = quantum_expectations(
            quantum_input,
            self.quantum_parameters,
        )

        quantum_raw_correction = float(
            np.dot(
                quantum_output,
                self.quantum_weights
            )
            + self.quantum_bias
        )

        quantum_correction = (
            self.quantum_alpha
            * quantum_raw_correction
        )

        # ----------------------------------------------------
        # Final V12 hybrid probability
        # ----------------------------------------------------

        hybrid_probability = float(
            sigmoid(
                logit(classical_probability)
                + quantum_correction
            )
        )

        classical_prediction = int(
            classical_probability
            >= CLASSICAL_THRESHOLD
        )

        hybrid_prediction = int(
            hybrid_probability
            >= HYBRID_THRESHOLD
        )

        explanation = self.explain(
            patient,
            classical_probability,
            hybrid_probability,
            quantum_correction,
        )

        return {
            "classical_probability": round(
                classical_probability * 100,
                2,
            ),

            "hybrid_probability": round(
                hybrid_probability * 100,
                2,
            ),

            "quantum_correction": round(
                quantum_correction,
                5,
            ),

            "classical_prediction":
                classical_prediction,

            "hybrid_prediction":
                hybrid_prediction,

            "classical_label":
                (
                    "Higher Risk Pattern"
                    if classical_prediction
                    else "Lower Risk Pattern"
                ),

            "hybrid_label":
                (
                    "Higher Risk Pattern"
                    if hybrid_prediction
                    else "Lower Risk Pattern"
                ),

            "model_agreement":
                classical_prediction == hybrid_prediction,

            "quantum_expectations":
                quantum_output.tolist(),

            "explanation":
                explanation,
        }

    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    def explain(
        self,
        patient,
        classical_probability,
        hybrid_probability,
        quantum_correction,
    ):
        """
        Lightweight model-specific explanation.

        We perturb each selected feature toward the training
        median and measure the change in the classical model
        probability.

        This is an explanation heuristic, not a clinical claim.
        """

        patient_df = pd.DataFrame(
            [patient]
        )

        X = pd.read_csv(
            self.dataset_path
        )

        explanations = []

        base_probability = classical_probability

        for feature in self.selected_features:

            modified = patient_df.copy()

            median_value = float(
                X[feature].median()
            )

            modified.loc[0, feature] = median_value

            scaled = self.classical_scaler.transform(
                modified[self.selected_features]
            )

            new_probability = float(
                self.classical_model.predict_proba(
                    scaled
                )[0, 1]
            )

            change = (
                base_probability
                - new_probability
            )

            explanations.append({
                "feature": feature,
                "probability_change": round(
                    change * 100,
                    3,
                ),
            })

        explanations.sort(
            key=lambda x: abs(
                x["probability_change"]
            ),
            reverse=True,
        )

        return {
            "top_features":
                explanations[:5],

            "quantum_effect":
                (
                    "increased"
                    if quantum_correction > 0
                    else "decreased"
                ),

            "quantum_effect_magnitude":
                round(
                    abs(quantum_correction),
                    5,
                ),

            "note":
                (
                    "Feature contributions are model "
                    "perturbation estimates and should "
                    "not be interpreted as clinical causation."
                ),
        }


# ============================================================
# TEST THE ENGINE DIRECTLY
# ============================================================

if __name__ == "__main__":

    print("\n==========================================")
    print("V12 HYBRID QML INFERENCE ENGINE")
    print("==========================================")

    model = V12HybridPredictor().load()

    # Example patient.
    # Replace these values with UI input later.
    example_patient = {

        "EducationLevel": 2,
        "SleepQuality": 5,
        "CardiovascularDisease": 0,
        "SystolicBP": 135,
        "CholesterolHDL": 50,
        "CholesterolTriglycerides": 150,
        "MMSE": 22,
        "FunctionalAssessment": 4,
        "MemoryComplaints": 1,
        "BehavioralProblems": 0,
        "ADL": 5,
        "PersonalityChanges": 0,
    }

    result = model.predict(
        example_patient
    )

    print("\n--------------- RESULT ---------------")

    print(
        f"Classical probability : "
        f"{result['classical_probability']}%"
    )

    print(
        f"Hybrid probability    : "
        f"{result['hybrid_probability']}%"
    )

    print(
        f"Quantum correction    : "
        f"{result['quantum_correction']}"
    )

    print(
        f"\nClassical prediction  : "
        f"{result['classical_label']}"
    )

    print(
        f"Hybrid prediction     : "
        f"{result['hybrid_label']}"
    )

    print(
        f"\nModels agree          : "
        f"{result['model_agreement']}"
    )

    print("\nTop explanatory features:")

    for item in result["explanation"]["top_features"]:

        print(
            f"  {item['feature']}: "
            f"{item['probability_change']:+.3f} percentage points"
        )

    print(
        "\nQuantum residual "
        f"{result['explanation']['quantum_effect']} "
        "the final probability."
    )

    print("\n==========================================")
    print("Research/decision-support prototype only.")
    print("Not a clinical diagnostic system.")
    print("==========================================")
