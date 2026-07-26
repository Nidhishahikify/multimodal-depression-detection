"""
scripts/train_model.py

KEY POINTS:
- Trains on train_audio.csv / train_image.csv
- Tests on test_audio.csv  / test_image.csv  (never seen during training)
- StandardScaler fitted ONLY on train data, then applied to test
- SMOTE only on training data (inside pipeline)
- Reports TRAIN accuracy and TEST accuracy separately
- TEST accuracy is your REAL accuracy number

Expected results with your dataset:
  Audio (600+600): ~70-80% test accuracy
  Image (2000+2000): ~72-82% test accuracy
"""
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    TRAIN_AUDIO_CSV, TEST_AUDIO_CSV,
    TRAIN_IMAGE_CSV, TEST_IMAGE_CSV,
    MODELS_DIR, RESULTS_DIR, LOGS_DIR,
    TRAIN_CONFIG, FUSION_WEIGHTS, LABEL_MAP,
)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler(LOGS_DIR / "train_model.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

RS = TRAIN_CONFIG["random_state"]
CV = TRAIN_CONFIG["cv_folds"]


# ── Load CSV ──────────────────────────────────────────────────────────────────
def load_csv(csv_path: Path, name: str):
    if not csv_path.exists():
        print(f"  [ERROR] File not found: {csv_path}")
        print(f"  Run extraction scripts first!")
        return None, None
    df = pd.read_csv(csv_path)
    y  = df["label"].values
    X  = df.drop(columns=["filename", "label"]).values.astype(np.float32)
    counts = {LABEL_MAP[k]: int(v)
              for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"    {name}: {X.shape[0]} samples x {X.shape[1]} features  {counts}")
    return X, y


# ── Build pipeline ────────────────────────────────────────────────────────────
def build_pipeline(n_pca=None):
    """
    Pipeline order:
      1. SMOTE       - oversample minority class (train only)
      2. StandardScaler - scale all features to mean=0, std=1
      3. PCA         - reduce dimensions (image only)
      4. VotingClassifier (RF + GBM + SVM soft vote)
    """
    steps = [
        ("smote",  SMOTE(random_state=RS,
                         k_neighbors=min(TRAIN_CONFIG["smote_k"], 3))),
        ("scaler", StandardScaler()),
    ]
    if n_pca:
        steps.append(("pca", PCA(n_components=n_pca, random_state=RS)))

    clf = VotingClassifier(
        estimators=[
            ("rf",  RandomForestClassifier(
                        n_estimators=200, max_depth=12,
                        class_weight="balanced",
                        random_state=RS, n_jobs=-1)),
            ("gbm", GradientBoostingClassifier(
                        n_estimators=150, learning_rate=0.05,
                        max_depth=4, random_state=RS)),
            ("svm", SVC(
                        kernel="rbf", probability=True,
                        class_weight="balanced",
                        C=1.0, gamma="scale", random_state=RS)),
        ],
        voting="soft",
        n_jobs=-1,
    )
    steps.append(("clf", clf))
    return ImbPipeline(steps)


# ── Print + save results ──────────────────────────────────────────────────────
def evaluate(y_true, y_pred, y_prob, modality: str, split: str) -> dict:
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec   = recall_score(y_true, y_pred,    average="weighted", zero_division=0)
    f1w   = f1_score(y_true, y_pred,        average="weighted", zero_division=0)
    f1m   = f1_score(y_true, y_pred,        average="macro",    zero_division=0)
    auc   = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)

    print(f"\n  {'='*50}")
    print(f"  {split} RESULTS -- {modality}")
    print(f"  {'='*50}")
    print(f"  Accuracy        : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision (w)   : {prec:.4f}")
    print(f"  Recall    (w)   : {rec:.4f}")
    print(f"  F1 (weighted)   : {f1w:.4f}")
    print(f"  F1 (macro)      : {f1m:.4f}")
    print(f"  ROC-AUC         : {auc:.4f}")
    print(f"  PR-AUC          : {prauc:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=list(LABEL_MAP.values()))}")

    # Save plots only for TEST set
    if split == "TEST":
        tag = modality.lower()

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix(y_true, y_pred),
            display_labels=list(LABEL_MAP.values())
        ).plot(ax=ax, cmap="Blues", colorbar=True)
        ax.set_title(f"Confusion Matrix - {modality} (TEST)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"confusion_matrix_{tag}.png", dpi=150)
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, color="#1f77b4", label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {modality} (TEST)", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"roc_curve_{tag}.png", dpi=150)
        plt.close()

        # PR curve
        p_arr, r_arr, _ = precision_recall_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(r_arr, p_arr, lw=2, color="#ff7f0e",
                label=f"PR-AUC = {prauc:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve - {modality} (TEST)", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"pr_curve_{tag}.png", dpi=150)
        plt.close()

        print(f"  Plots saved to results/")

    return {
        "accuracy"    : round(acc,   4),
        "precision_w" : round(prec,  4),
        "recall_w"    : round(rec,   4),
        "f1_weighted" : round(f1w,   4),
        "f1_macro"    : round(f1m,   4),
        "roc_auc"     : round(auc,   4),
        "pr_auc"      : round(prauc, 4),
    }


# ── Train one modality ────────────────────────────────────────────────────────
def train_modality(train_csv: Path, test_csv: Path,
                   modality: str, n_pca=None):
    print(f"\n{'='*55}")
    print(f"  TRAINING {modality.upper()} MODEL")
    print(f"{'='*55}")
    print("  Loading data...")

    X_train, y_train = load_csv(train_csv, "TRAIN")
    X_test,  y_test  = load_csv(test_csv,  "TEST")

    if X_train is None or X_test is None:
        print(f"  Skipping {modality} - data not found.")
        return None, {}

    # --- Cross-validation on training data only ---
    print(f"\n  Running {CV}-fold cross-validation on training data...")
    pipe_cv = build_pipeline(n_pca)
    skf     = StratifiedKFold(n_splits=CV, shuffle=True, random_state=RS)
    cv_res  = cross_validate(
        pipe_cv, X_train, y_train,
        cv=skf,
        scoring={"accuracy": "accuracy",
                 "f1_weighted": "f1_weighted",
                 "roc_auc": "roc_auc"},
    )
    cv_acc = np.mean(cv_res["test_accuracy"])
    cv_f1  = np.mean(cv_res["test_f1_weighted"])
    cv_auc = np.mean(cv_res["test_roc_auc"])
    print(f"  CV Accuracy : {cv_acc:.4f} +/- {np.std(cv_res['test_accuracy']):.4f}")
    print(f"  CV F1       : {cv_f1:.4f} +/- {np.std(cv_res['test_f1_weighted']):.4f}")
    print(f"  CV ROC-AUC  : {cv_auc:.4f} +/- {np.std(cv_res['test_roc_auc']):.4f}")

    # --- Train final model on ALL training data ---
    print(f"\n  Training final model on all {len(X_train)} training samples...")
    pipe = build_pipeline(n_pca)
    pipe.fit(X_train, y_train)

    # --- Training accuracy (how well model learned) ---
    y_pred_tr = pipe.predict(X_train)
    y_prob_tr = pipe.predict_proba(X_train)[:, 1]
    train_m   = evaluate(y_train, y_pred_tr, y_prob_tr, modality, "TRAIN")

    # --- TEST accuracy (REAL accuracy - unseen data) ---
    print(f"\n  Evaluating on {len(X_test)} UNSEEN test samples...")
    y_pred_te = pipe.predict(X_test)
    y_prob_te = pipe.predict_proba(X_test)[:, 1]
    test_m    = evaluate(y_test, y_pred_te, y_prob_te, modality, "TEST")

    # Save model
    model_path = MODELS_DIR / f"model_{modality.lower()}.pkl"
    joblib.dump(pipe, model_path)
    print(f"  Model saved -> {model_path.name}")

    # Save metrics JSON
    metrics = {
        "cv_accuracy_mean" : round(float(cv_acc), 4),
        "cv_f1_mean"       : round(float(cv_f1),  4),
        "cv_roc_auc_mean"  : round(float(cv_auc), 4),
        "train"            : train_m,
        "test"             : test_m,
    }
    with open(RESULTS_DIR / f"metrics_{modality.lower()}.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    return pipe, metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", action="store_true", help="Train audio only")
    parser.add_argument("--image", action="store_true", help="Train image only")
    args = parser.parse_args()

    do_audio = args.audio or (not args.audio and not args.image)
    do_image = args.image or (not args.audio and not args.image)

    all_metrics  = {}
    audio_pipe   = None
    image_pipe   = None

    if do_audio:
        audio_pipe, m = train_modality(
            TRAIN_AUDIO_CSV, TEST_AUDIO_CSV, "Audio")
        if m:
            all_metrics["Audio"] = m

    if do_image:
        image_pipe, m = train_modality(
            TRAIN_IMAGE_CSV, TEST_IMAGE_CSV, "Image", n_pca=100)
        if m:
            all_metrics["Image"] = m

    # Fusion (index-based since audio+image are different people)
    if audio_pipe is not None and image_pipe is not None:
        print(f"\n{'='*55}")
        print("  FUSION - TEST SET (index-based, separate datasets)")
        print(f"{'='*55}")
        X_at, y_at = load_csv(TEST_AUDIO_CSV, "TEST-Audio")
        X_it, y_it = load_csv(TEST_IMAGE_CSV, "TEST-Image")

        if X_at is not None and X_it is not None:
            # Use the smaller test set size
            n = min(len(y_at), len(y_it))
            p_a = audio_pipe.predict_proba(X_at[:n])[:, 1]
            p_i = image_pipe.predict_proba(X_it[:n])[:, 1]
            p_f = FUSION_WEIGHTS["audio"] * p_a + FUSION_WEIGHTS["image"] * p_i
            y_f = (p_f >= 0.5).astype(int)

            # Use audio labels as ground truth
            fuse_m = evaluate(y_at[:n], y_f, p_f, "Fusion", "TEST")
            all_metrics["Fusion"] = {"test": fuse_m}

            # Save fusion model
            joblib.dump(
                {"audio": audio_pipe, "image": image_pipe,
                 "weights": FUSION_WEIGHTS},
                MODELS_DIR / "model_fusion.pkl",
            )
            print(f"  Fusion model saved -> model_fusion.pkl")

    # ── Final summary table ────────────────────────────────────────────────────
    if all_metrics:
        print(f"\n{'='*55}")
        print("  FINAL ACCURACY SUMMARY (TEST SET)")
        print(f"  This is your real accuracy on unseen data")
        print(f"{'='*55}")
        rows = []
        for mod, m in all_metrics.items():
            t = m.get("test", {})
            rows.append({
                "Modality"  : mod,
                "Accuracy"  : f"{t.get('accuracy', 0)*100:.2f}%",
                "F1(w)"     : f"{t.get('f1_weighted', 0):.4f}",
                "F1(macro)" : f"{t.get('f1_macro', 0):.4f}",
                "ROC-AUC"   : f"{t.get('roc_auc', 0):.4f}",
                "PR-AUC"    : f"{t.get('pr_auc', 0):.4f}",
            })
        df = pd.DataFrame(rows).set_index("Modality")
        print(f"\n{df.to_string()}\n")
        df.to_csv(RESULTS_DIR / "accuracy_summary.csv")
        print(f"  Saved -> results/accuracy_summary.csv")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()