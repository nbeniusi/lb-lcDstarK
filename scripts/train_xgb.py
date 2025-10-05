#!/usr/bin/env python3
"""
Train an XGBoost classifier for signal-vs-background with proper weighting,
no data leakage, and clear evaluation (ROC/AUC, PR/AP).

Inputs (defaults match Step 3 outputs):
  data/processed/features_bg.parquet
  data/processed/features_sg_weighted_mLcDst.parquet

Outputs:
  - figures/roc_xgb.png, figures/pr_xgb.png
  - outputs/models/xgb_mLcDst.json (trained model)
  - outputs/models/run.json (manifest: params, features, metrics, files)

Usage:
  python scripts/train_xgb.py \
    --bg data/processed/features_bg.parquet \
    --sg data/processed/features_sg_weighted_mLcDst.parquet \
    --out-model outputs/models/xgb_mLcDst.json \
    --figdir figures
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, roc_auc_score, PrecisionRecallDisplay, average_precision_score
)
from xgboost import XGBClassifier


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_parquets(bg_path, sg_path):
    bg = pd.read_parquet(bg_path)
    sg = pd.read_parquet(sg_path)

    # Ensure required columns exist
    for col in ["label"]:
        if col not in bg.columns or col not in sg.columns:
            raise SystemExit(f"[ERROR] expected column '{col}' missing in bg or sg parquet.")

    # Background has no 'weight' → assign 1.0
    if "weight" not in bg.columns:
        bg["weight"] = 1.0
    # Signal already has 'weight' from Step 2 (Breit–Wigner reweighting)
    if "weight" not in sg.columns:
        sg["weight"] = 1.0

    return bg, sg


def build_feature_list(df, drop_res_mass=False):
    """
    Default features: three masses, their squares, and helicity angles if present.
    Optionally drop the resonant mass mLcDst (+ its square) for 'mass-drop' test.
    """
    base = ["mLcK","mDstK","mLcDst","mLcK2","mDstK2","mLcDst2"]
    angles = [c for c in ["costh_LcK","costh_DstK","costh_LcDst"] if c in df.columns]
    feats = base + angles
    if drop_res_mass:
        for c in ["mLcDst","mLcDst2"]:
            if c in feats:
                feats.remove(c)
    return feats


def main():
    ap = argparse.ArgumentParser(description="Train XGBoost with weighted signal and proper splits.")
    ap.add_argument("--bg", default="data/processed/features_bg.parquet", help="Background parquet")
    ap.add_argument("--sg", default="data/processed/features_sg_weighted_mLcDst.parquet", help="Signal weighted parquet")
    ap.add_argument("--out-model", default="outputs/models/xgb_mLcDst.json", help="Where to save the model (.json)")
    ap.add_argument("--figdir", default="figures", help="Where to save figures")
    ap.add_argument("--drop-resonant-mass", action="store_true",
                    help="Exclude mLcDst & mLcDst2 (mass-drop test)")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test fraction (e.g., 0.2)")
    ap.add_argument("--valid-size", type=float, default=0.2, help="Validation fraction *of the TRAIN* (e.g., 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out_model))
    ensure_dir(args.figdir)

    # 1) Load and combine
    bg, sg = load_parquets(args.bg, args.sg)
    df_all = pd.concat([bg, sg], ignore_index=True)

    # 2) Build feature matrix / labels / sample weights
    features = build_feature_list(df_all, drop_res_mass=args.drop_resonant_mass)
    missing = [f for f in features if f not in df_all.columns]
    if missing:
        raise SystemExit(f"[ERROR] missing expected feature columns: {missing}")

    X = df_all[features].to_numpy()
    y = df_all["label"].to_numpy().astype(int)
    w = df_all["weight"].to_numpy().astype(float)

    # --- Balance total weight of classes: sum_pos == sum_neg
    # Negative class (label 0) has weight ~1 per event
    neg_mask = (y == 0)
    pos_mask = (y == 1)
    sum_neg = float(w[neg_mask].sum())
    sum_pos = float(w[pos_mask].sum())

    if sum_pos <= 0:
        raise SystemExit("[ERROR] Positive (signal) total weight is zero. Check your weight file and merging.")

    alpha = sum_neg / sum_pos
    w[pos_mask] *= alpha
    # Also balance validation/test weights consistently:
    # (we will recompute w_train/w_valid/w_test after the split below, so this scaling applies everywhere)



    # 3) Stratified split → (train+valid) and test  (NO leakage)
    X_trv, X_test, y_trv, y_test, w_trv, w_test = train_test_split(
        X, y, w, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # 4) Validation split from training portion only (for early stopping)
    X_train, X_valid, y_train, y_valid, w_train, w_valid = train_test_split(
        X_trv, y_trv, w_trv, test_size=args.valid_size, stratify=y_trv, random_state=args.seed
    )

    # 5) Model (robust baseline)
    clf = XGBClassifier(
        n_estimators=2000,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.8,
        eval_metric="auc",
        n_jobs=-1,
        random_state=args.seed,
    )

        # 6) Train with early stopping on VALIDATION set (version-robust)
    try:
        # Newer xgboost: supports early_stopping_rounds + sample_weight_eval_set
        clf.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_valid, y_valid)],
            sample_weight_eval_set=[w_valid],
            verbose=False,
            early_stopping_rounds=100
        )
    except TypeError:
        # Older xgboost: no early_stopping_rounds in sklearn API
        print("[WARN] xgboost.fit() does not accept 'early_stopping_rounds' — trying callbacks fallback.")
        try:
            from xgboost.callback import EarlyStopping
            clf.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
                callbacks=[EarlyStopping(rounds=100, save_best=True, maximize=True)]
            )
        except Exception:
            print("[WARN] EarlyStopping callback not available — training without early stopping.")
            # Use a conservative number of trees
            clf.set_params(n_estimators=600)
            clf.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False
            )


    # 7) Evaluate on HELD-OUT TEST set (final numbers)
    proba_test = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test, sample_weight=w_test)

    # ROC curve
    fpr, tpr, thr = roc_curve(y_test, proba_test, sample_weight=w_test)
    plt.figure(figsize=(5.8,4.6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0,1], [0,1], "k--", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    fig_roc = os.path.join(args.figdir, "roc_xgb.png")
    plt.savefig(fig_roc, dpi=300)
    plt.close()

    # Precision–Recall
    ap = average_precision_score(y_test, proba_test, sample_weight=w_test)
    disp = PrecisionRecallDisplay.from_predictions(y_test, proba_test, sample_weight=w_test)
    plt.gca().set_title(f"Average precision = {ap:.3f}")
    plt.tight_layout()
    fig_pr = os.path.join(args.figdir, "pr_xgb.png")
    plt.savefig(fig_pr, dpi=300)
    plt.close()

    # 8) Save model + manifest (reproducibility)
    clf.save_model(args.out_model)
    manifest = {
        "inputs": {"bg": args.bg, "sg": args.sg},
        "features": features,
        "drop_resonant_mass": bool(args.drop_resonant_mass),
        "splits": {"test_size": args.test_size, "valid_size": args.valid_size, "seed": args.seed},
        "xgb_params": {
            "n_estimators": int(clf.n_estimators),
            "max_depth": int(clf.max_depth),
            "learning_rate": float(clf.learning_rate),
            "subsample": float(clf.subsample),
            "colsample_bytree": float(clf.colsample_bytree),
            "eval_metric": "auc"
        },
        "metrics": {"AUC": float(auc), "AP": float(ap)},
        "figures": {"roc": fig_roc, "pr": fig_pr},
        "model_path": args.out_model
    }
    with open(os.path.join(os.path.dirname(args.out_model), "run.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n[SUMMARY]")
    print(f"  Features      : {features}")
    print(f"  AUC (test)    : {auc:.4f}")
    print(f"  AP  (test)    : {ap:.4f}")
    print(f"  ROC figure    : {fig_roc}")
    print(f"  PR  figure    : {fig_pr}")
    print(f"  Model saved   : {args.out_model}")
    print(f"  Manifest      : outputs/models/run.json\n")


if __name__ == "__main__":
    main()
