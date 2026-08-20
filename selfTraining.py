"""
ATSC 3.0 Update Broadcast Agent — Self-Training (Decision Tree base, per-parameter, threshold criterion)
========================================================================================================
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from dataGeneration import (
    PLP_OPTIONS, MODULATION_OPTIONS, CODE_RATE_OPTIONS, LDPC_LENGTH_OPTIONS,
    CRITICAL_FLOORS,
)

np.random.seed(42)

DATA_DIR = "data"
MODEL_DIR = "models"
PLOT_DIR = "shap_plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

TARGET_COLUMNS = [
    "plp", "modulation", "code_rate", "ldpc_length",
    "al_fec_redundancy_pct", "carousel_repeat_sec",
]
NUMERIC_FEATURES = ["priority", "size_mb"]
CATEGORICAL_FEATURES = ["update_type", "target_ecu"]

FULL_CLASS_LISTS = {
    "plp": PLP_OPTIONS,
    "modulation": MODULATION_OPTIONS,
    "code_rate": CODE_RATE_OPTIONS,
    "ldpc_length": LDPC_LENGTH_OPTIONS,
    "al_fec_redundancy_pct": ["0", "5", "15", "30", "50"],
    "carousel_repeat_sec": ["0", "120", "300", "900", "1800"],
}

CONFIDENCE_THRESHOLD = 0.75  # standard default -- only add predictions the
                              # tree is at least this confident about


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
labeled_seed = pd.read_csv(f"{DATA_DIR}/labeled_seed.csv")
unlabeled_pool = pd.read_csv(f"{DATA_DIR}/unlabeled_pool.csv")
labeled_test = pd.read_csv(f"{DATA_DIR}/labeled_test.csv")

print(f"Labeled seed:   {len(labeled_seed)} rows")
print(f"Unlabeled pool: {len(unlabeled_pool)} rows")
print(f"Held-out test:  {len(labeled_test)} rows")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")


# ---------------------------------------------------------------------------
# FEATURE ENCODING
# ---------------------------------------------------------------------------
fit_pool = pd.concat([labeled_seed, unlabeled_pool], ignore_index=True)
scaler = StandardScaler().fit(fit_pool[NUMERIC_FEATURES])
onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
    fit_pool[CATEGORICAL_FEATURES]
)
feature_names = NUMERIC_FEATURES + list(onehot.get_feature_names_out(CATEGORICAL_FEATURES))


def build_features(df: pd.DataFrame) -> np.ndarray:
    num = scaler.transform(df[NUMERIC_FEATURES])
    cat = onehot.transform(df[CATEGORICAL_FEATURES])
    return np.hstack([num, cat])


X_labeled = build_features(labeled_seed)
X_unlabeled = build_features(unlabeled_pool)
X_test = build_features(labeled_test)
X_combined = np.vstack([X_labeled, X_unlabeled])


# ---------------------------------------------------------------------------
# SELF-TRAINING (Decision Tree base, threshold criterion), one per parameter
# ---------------------------------------------------------------------------
self_trained_models = {}
label_encoders = {}
accuracies = {}
pool_usage = {}

for target_col in TARGET_COLUMNS:
    print(f"\n=== {target_col} ===")

    le = LabelEncoder().fit(FULL_CLASS_LISTS[target_col])
    label_encoders[target_col] = le

    y_labeled = le.transform(labeled_seed[target_col].astype(str))
    y_unlabeled = np.full(len(unlabeled_pool), -1)
    y_combined = np.concatenate([y_labeled, y_unlabeled])

    base_tree = DecisionTreeClassifier(max_depth=6, random_state=42)
    self_trainer = SelfTrainingClassifier(
        estimator=base_tree,
        criterion="threshold",
        threshold=CONFIDENCE_THRESHOLD,
        max_iter=20,
        verbose=False,
    )
    self_trainer.fit(X_combined, y_combined)
    self_trained_models[target_col] = self_trainer

    pool_iters = self_trainer.labeled_iter_[len(labeled_seed):]
    used = np.sum(pool_iters != -1)
    pool_usage[target_col] = (used, len(unlabeled_pool))
    print(f"Unlabeled pool used: {used}/{len(unlabeled_pool)} "
          f"({100 * used / len(unlabeled_pool):.1f}%)")

    y_test_true = le.transform(labeled_test[target_col].astype(str))
    y_test_pred = self_trainer.predict(X_test)
    acc = accuracy_score(y_test_true, y_test_pred)
    accuracies[target_col] = acc
    print(f"Held-out test accuracy: {acc:.3f}")

    joblib.dump(self_trainer, f"{MODEL_DIR}/selftrain_tree_{target_col}.joblib")
    joblib.dump(le, f"{MODEL_DIR}/label_encoder_{target_col}.joblib")

    # --- SHAP directly on the fitted Decision Tree ---
    fitted_tree = self_trainer.estimator_
    explainer = shap.TreeExplainer(fitted_tree)
    sample = X_test[: min(200, len(X_test))]
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        plot_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))
        plot_values = np.abs(shap_values).mean(axis=2)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)
        plot_values = shap_values

    ranking = sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])
    print("Top features driving this decision (mean |SHAP value|):")
    for name, val in ranking[:5]:
        print(f"  {name:25s} {val:.4f}")

    plt.figure()
    shap.summary_plot(plot_values, sample, feature_names=feature_names,
                       plot_type="bar", show=False)
    plt.title(f"SHAP feature importance — {target_col} (tree, threshold)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/shap_tree_threshold_{target_col}.png", dpi=120)
    plt.close()


print("\n" + "=" * 60)
print(f"Accuracy summary (held-out test set, threshold={CONFIDENCE_THRESHOLD}):")
for col, acc in accuracies.items():
    used, total = pool_usage[col]
    print(f"  {col:25s} acc={acc:.3f}   pool used={used}/{total} ({100*used/total:.0f}%)")


# ---------------------------------------------------------------------------
# SAFETY FLOOR ENFORCEMENT
# ---------------------------------------------------------------------------

def enforce_safety_floors(update_type: str, prediction: dict) -> dict:
    if update_type != "critical":
        return prediction
    p = dict(prediction)
    if MODULATION_OPTIONS.index(p["modulation"]) > MODULATION_OPTIONS.index(CRITICAL_FLOORS["modulation"]):
        p["modulation"] = CRITICAL_FLOORS["modulation"]
    if CODE_RATE_OPTIONS.index(p["code_rate"]) > CODE_RATE_OPTIONS.index(CRITICAL_FLOORS["code_rate"]):
        p["code_rate"] = CRITICAL_FLOORS["code_rate"]
    if p["ldpc_length"] != CRITICAL_FLOORS["ldpc_length"]:
        p["ldpc_length"] = CRITICAL_FLOORS["ldpc_length"]
    if p["al_fec_redundancy_pct"] < CRITICAL_FLOORS["al_fec_redundancy_pct"]:
        p["al_fec_redundancy_pct"] = CRITICAL_FLOORS["al_fec_redundancy_pct"]
    if p["carousel_repeat_sec"] == 0 or p["carousel_repeat_sec"] > CRITICAL_FLOORS["carousel_repeat_sec"]:
        p["carousel_repeat_sec"] = CRITICAL_FLOORS["carousel_repeat_sec"]
    return p


def predict_config(update_type: str, target_ecu: str, priority: int, size_mb: float) -> dict:
    row = pd.DataFrame([{
        "update_type": update_type, "target_ecu": target_ecu,
        "priority": priority, "size_mb": size_mb,
    }])
    X = build_features(row)
    prediction = {}
    for col in TARGET_COLUMNS:
        pred_encoded = self_trained_models[col].predict(X)[0]
        prediction[col] = label_encoders[col].inverse_transform([pred_encoded])[0]
    prediction["al_fec_redundancy_pct"] = int(prediction["al_fec_redundancy_pct"])
    prediction["carousel_repeat_sec"] = int(prediction["carousel_repeat_sec"])
    return enforce_safety_floors(update_type, prediction)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DEMO: predicting config for a sample update")
    demo = predict_config(update_type="critical", target_ecu="braking", priority=5, size_mb=12.0)
    print("Critical braking update, priority 5, 12MB ->")
    for k, v in demo.items():
        print(f"  {k:25s} {v}")

    demo2 = predict_config(update_type="non_critical", target_ecu="infotainment", priority=1, size_mb=200.0)
    print("\nNon-critical infotainment update, priority 1, 200MB ->")
    for k, v in demo2.items():
        print(f"  {k:25s} {v}")
