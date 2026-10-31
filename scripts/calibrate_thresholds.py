"""
scripts/calibrate_thresholds.py

Derives a validated decision threshold SEPARATELY for the audio and image
models and writes them to thresholds.json.

Method (no test-set leakage):
  1. Load the saved trained model and the TRAIN feature CSV.
  2. Get out-of-fold (OOF) scores on the train set with stratified K-fold
     (a fresh clone of the model is fitted on each fold, so every score
     comes from a model that never saw that sample).
  3. Check calibration (Brier score, ECE). Optionally fit a Platt sigmoid
     on the OOF scores (auto = only if Brier improves by >= 5%).
  4. Pick the threshold on the OOF scores using the chosen strategy
     (default: maximise F-beta with beta=2, i.e. recall-weighted, because a
     missed indicator is costlier than a false alarm in a screening tool).
  5. Evaluate ONCE on the held-out TEST CSV with the already-trained model
     and report precision / recall / F1 / ROC-AUC / confusion matrix,
     also compared against the naive 0.5 cutoff.

Usage (from project root):
    py -3.11 scripts/calibrate_thresholds.py                 # both modalities
    py -3.11 scripts/calibrate_thresholds.py --modality audio
    py -3.11 scripts/calibrate_thresholds.py --strategy recall --min-recall 0.85
    py -3.11 scripts/calibrate_thresholds.py --strategy youden
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict)

import config as C
from threshold_utils import apply_calibration, get_scores, sigmoid

warnings.filterwarnings("ignore")

LABEL_CANDIDATES = ["label", "Label", "labels", "class", "Class", "target", "y", "depressed"]
GROUP_CANDIDATES = ["subject", "speaker", "person", "participant", "patient", "id", "group"]


# ── Data loading ─────────────────────────────────────────────────────────────
def map_labels(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).to_numpy()
    out = []
    for v in series.astype(str).str.lower():
        out.append(0 if ("non" in v or v in ("0", "normal", "healthy")) else 1)
    return np.array(out)


def load_xy(csv_path, n_expected, label_col=None, group_col=None):
    df = pd.read_csv(csv_path)
    lc = label_col or next((c for c in LABEL_CANDIDATES if c in df.columns), df.columns[-1])
    y = map_labels(df[lc])
    groups = None
    if group_col is None:
        group_col = next((c for c in GROUP_CANDIDATES if c in df.columns and c != lc), None)
    if group_col:
        groups = df[group_col].to_numpy()
    drop = {lc} | ({group_col} if group_col else set())
    feats = df.drop(columns=[c for c in drop if c in df.columns])
    X = feats.select_dtypes(include=[np.number])
    if X.shape[1] != n_expected:
        raise SystemExit(
            f"[ERROR] {csv_path.name}: {X.shape[1]} numeric feature columns but the "
            f"model expects {n_expected}. Non-feature columns found: "
            f"{[c for c in feats.columns if c not in X.columns]}. "
            f"Use --label-col / --group-col or clean the CSV."
        )
    return X.to_numpy(dtype=float), y, groups, lc, group_col


# ── Metrics ──────────────────────────────────────────────────────────────────
def metrics_at(y, s, t):
    pred = (s >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0.0
    return {
        "threshold": round(float(t), 4),
        "accuracy": round((tp + tn) / len(y), 4),
        "balanced_accuracy": round((rec + spec) / 2, 4),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "specificity": round(spec, 4), "f1": round(f1, 4), "f2": round(f2, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def ece_score(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


# ── Threshold selection ──────────────────────────────────────────────────────
def candidate_thresholds(scores, kind):
    if kind == "probability":
        return np.linspace(0.05, 0.95, 181)
    return np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 193)))


def pick_threshold(y, s, kind, strategy, beta, min_recall, min_spec):
    """
    Choose a threshold on out-of-fold scores.
    A specificity floor (min_spec) stops recall-heavy strategies from
    degenerating into "flag everyone as depressed" when the model is weak.
    """
    cands = candidate_thresholds(s, kind)
    ms = [metrics_at(y, s, t) for t in cands]
    spec = np.array([m["specificity"] for m in ms])
    obj = np.zeros(len(cands))
    for i, m in enumerate(ms):
        p, r = m["precision"], m["recall"]
        if strategy == "fbeta":
            b2 = beta ** 2
            obj[i] = (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0
        elif strategy == "youden":
            obj[i] = r + m["specificity"] - 1
        elif strategy == "recall":
            obj[i] = p if r >= min_recall else -1.0
    allowed = spec >= min_spec
    if not allowed.any():
        print(f"   [WARN] no threshold reaches specificity >= {min_spec}; "
              "model is too weak for this constraint - falling back to Youden's J")
        obj = np.array([m["recall"] + m["specificity"] - 1 for m in ms])
        allowed = np.ones(len(cands), dtype=bool)
    obj = np.where(allowed, obj, -np.inf)
    if strategy == "recall" and obj.max() < 0:
        print(f"   [WARN] no allowed threshold reaches recall >= {min_recall}; using Youden's J")
        obj = np.where(allowed, [m["recall"] + m["specificity"] - 1 for m in ms], -np.inf)
    best = obj.max()
    ok = obj >= best - 0.01 * abs(best)   # plateau smoothing: median of near-best thresholds
    return float(np.median(cands[ok])), float(best)


# ── Per-modality pipeline ────────────────────────────────────────────────────
def run_modality(name, model_path, train_csv, test_csv, args):
    print(f"\n{'=' * 60}\n  {name.upper()} MODEL\n{'=' * 60}")
    if not Path(model_path).exists():
        print(f"  [SKIP] model not found: {model_path}")
        return None
    model = joblib.load(model_path)
    n_in = getattr(model, "n_features_in_", None)
    print(f"  model type      : {type(model).__name__}")
    print(f"  expects features: {n_in}")
    _, kind = get_scores(model, np.zeros((1, n_in or 1)))  # dry run for kind only
    print(f"  score type      : {kind}")
    if kind == "none":
        print("  This model only returns hard class labels, so no meaningful "
              "threshold can be derived. Retrain with a probabilistic classifier.")
        return {"score_type": "none", "note": "model gives hard labels only"}

    X, y, groups, lc, gc = load_xy(Path(train_csv), n_in, args.label_col, args.group_col)
    print(f"  train samples   : {len(y)}  (positives={int(y.sum())}, negatives={int((1 - y).sum())})")
    print(f"  label column    : {lc} | group column: {gc}")
    if gc is None:
        print("  [NOTE] no subject/group column -> if one person contributes several "
              "files/frames, folds can leak and scores may be optimistic.")

    # 1) out-of-fold scores ---------------------------------------------------
    method = "predict_proba" if kind == "probability" else "decision_function"
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=args.cv, shuffle=True, random_state=42)
        splits = list(cv.split(X, y, groups))
    else:
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
        splits = list(cv.split(X, y))
    print(f"  computing {args.cv}-fold out-of-fold scores (this can take a while)...")
    oof = cross_val_predict(clone(model), X, y, cv=splits, method=method)
    oof = oof[:, 1] if oof.ndim == 2 else oof

    # 2) calibration ----------------------------------------------------------
    calibration, s_use = None, oof
    cal_info = {}
    if kind == "probability":
        brier_raw, ece_raw = brier_score_loss(y, oof), ece_score(y, oof)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(oof.reshape(-1, 1), y)
        a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
        cal_scores = 1 / (1 + np.exp(-(a * oof + b)))
        brier_cal, ece_cal = brier_score_loss(y, cal_scores), ece_score(y, cal_scores)
        use_cal = args.calibrate == "on" or (args.calibrate == "auto" and brier_cal < 0.95 * brier_raw)
        cal_info = {"brier_raw": round(brier_raw, 4), "brier_platt": round(brier_cal, 4),
                    "ece_raw": round(ece_raw, 4), "ece_platt": round(ece_cal, 4),
                    "platt_applied": bool(use_cal)}
        print(f"  calibration     : Brier raw={brier_raw:.4f} platt={brier_cal:.4f} | "
              f"ECE raw={ece_raw:.4f} platt={ece_cal:.4f} -> "
              f"{'APPLY Platt' if use_cal else 'keep raw scores'}")
        if use_cal:
            calibration = {"type": "platt", "a": round(a, 6), "b": round(b, 6)}
            s_use = cal_scores
    else:
        print("  [NOTE] scores are decision values, NOT probabilities; calibration skipped.")

    # 3) pick threshold on OOF scores ------------------------------------------
    thr, obj = pick_threshold(y, s_use, kind, args.strategy, args.beta, args.min_recall, args.min_specificity)
    oof_auc = roc_auc_score(y, s_use)
    oof_m = metrics_at(y, s_use, thr)
    if oof_auc < 0.65:
        print(f"  [WARN] OOF ROC-AUC={oof_auc:.2f} is weak: the {name} model separates the classes only "
              "slightly, so any threshold will be a poor trade-off. Report this honestly.")
    print(f"  OOF ROC-AUC     : {oof_auc:.4f}")
    print(f"  chosen threshold: {thr:.4f}  (strategy={args.strategy})")
    print(f"  OOF @thr        : P={oof_m['precision']} R={oof_m['recall']} F1={oof_m['f1']} "
          f"CM={oof_m['confusion_matrix']}")

    result = {
        "threshold": round(thr, 4),
        "score_type": kind,
        "calibration": calibration,
        "strategy": args.strategy,
        "strategy_params": {"beta": args.beta, "min_recall": args.min_recall,
                            "min_specificity": args.min_specificity},
        "n_train": int(len(y)),
        "cv_folds": args.cv,
        "grouped_cv": groups is not None,
        "oof_roc_auc": round(float(oof_auc), 4),
        "oof_metrics_at_threshold": oof_m,
        "calibration_check": cal_info,
    }

    # 4) one-time held-out test evaluation ------------------------------------
    if test_csv and Path(test_csv).exists():
        Xt, yt, _, _, _ = load_xy(Path(test_csv), n_in, args.label_col, args.group_col)
        st, _ = get_scores(model, Xt)
        if calibration:
            st = np.array([apply_calibration(v, calibration) for v in st])
        test = {
            "n_test": int(len(yt)),
            "roc_auc": round(float(roc_auc_score(yt, st)), 4),
            "average_precision": round(float(average_precision_score(yt, st)), 4),
            "at_chosen_threshold": metrics_at(yt, st, thr),
            "at_default_0.5": metrics_at(yt, st, 0.5 if kind == "probability" else 0.0),
        }
        if kind == "probability":
            test["brier"] = round(float(brier_score_loss(yt, st)), 4)
        result["test_metrics"] = test
        m, d = test["at_chosen_threshold"], test["at_default_0.5"]
        print(f"  TEST ROC-AUC    : {test['roc_auc']}")
        print(f"  TEST @chosen    : P={m['precision']} R={m['recall']} F1={m['f1']} CM={m['confusion_matrix']}")
        print(f"  TEST @0.5       : P={d['precision']} R={d['recall']} F1={d['f1']} CM={d['confusion_matrix']}")
    else:
        print("  [NOTE] test CSV not found - skipped held-out evaluation")

    make_plots(name, y, s_use, thr, kind)
    return result


def make_plots(name, y, s, thr, kind):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import precision_recall_curve, roc_curve
        fig, ax = plt.subplots(1, 3 if kind == "probability" else 2, figsize=(14 if kind == "probability" else 10, 4))
        fpr, tpr, _ = roc_curve(y, s)
        ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--", c="gray")
        ax[0].set(title=f"{name} ROC (OOF)", xlabel="FPR", ylabel="TPR")
        p, r, _ = precision_recall_curve(y, s)
        ax[1].plot(r, p); ax[1].set(title=f"{name} Precision-Recall (OOF)", xlabel="Recall", ylabel="Precision")
        m = metrics_at(y, s, thr)
        ax[0].scatter([1 - m["specificity"]], [m["recall"]], c="red", zorder=5, label=f"thr={thr:.2f}")
        ax[0].legend(); ax[1].scatter([m["recall"]], [m["precision"]], c="red", zorder=5)
        if kind == "probability":
            fp_, mp_ = calibration_curve(y, s, n_bins=10, strategy="quantile")
            ax[2].plot(mp_, fp_, "o-"); ax[2].plot([0, 1], [0, 1], "--", c="gray")
            ax[2].set(title=f"{name} reliability", xlabel="Mean score", ylabel="Fraction positive")
        fig.tight_layout()
        C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = C.RESULTS_DIR / f"threshold_{name}.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"  plot saved      : {out}")
    except Exception as e:  # plots are optional
        print(f"  [WARN] plot skipped: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["audio", "image", "both"], default="both")
    ap.add_argument("--strategy", choices=["fbeta", "youden", "recall"], default="fbeta")
    ap.add_argument("--beta", type=float, default=2.0, help="F-beta (2 favours recall)")
    ap.add_argument("--min-recall", type=float, default=0.85, help="used by --strategy recall")
    ap.add_argument("--min-specificity", type=float, default=0.5,
                    help="floor on specificity so the threshold cannot flag everyone")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--calibrate", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--group-col", default=None, help="subject/speaker column, if your CSV has one")
    ap.add_argument("--output", default=str(C.THRESHOLDS_FILE))
    args = ap.parse_args()

    out_path = Path(args.output)
    cfg = {}
    if out_path.exists():
        try:
            cfg = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}

    jobs = {
        "audio": (C.MODELS_DIR / "model_audio.pkl", C.TRAIN_AUDIO_CSV, C.TEST_AUDIO_CSV),
        "image": (C.MODELS_DIR / "model_image.pkl", C.TRAIN_IMAGE_CSV, C.TEST_IMAGE_CSV),
    }
    for name, (mp, tr, te) in jobs.items():
        if args.modality in (name, "both"):
            res = run_modality(name, mp, tr, te, args)
            if res is not None:
                cfg[name] = res

    cfg["fusion"] = {
        "method": "threshold-normalised weighted average (each modality threshold -> 0.5)",
        "weights": C.FUSION_WEIGHTS,
        "threshold": 0.5,
        "validated_on_paired_data": False,
        "note": "Audio and image datasets are from different people, so fusion cannot "
                "be validated on matched pairs; treat it as an unvalidated late-fusion rule.",
    }
    cfg["meta"] = {"generated": datetime.now().isoformat(timespec="seconds"),
                   "disclaimer": "Experimental screening prototype; not a diagnostic tool."}
    out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\nSaved thresholds -> {out_path}")


if __name__ == "__main__":
    main()