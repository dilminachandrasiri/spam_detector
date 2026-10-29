"""
train_model.py - the AI / ML part of the Spam Detector.

Pipeline
--------
  1. Load the SMS Spam Collection CSV (columns v1 = label, v2 = text, rest = junk)
  2. Clean: drop empty rows + exact duplicate messages (they leak between train/test)
  3. Stratified 80/20 train/test split (spam is only ~13% of the data)
  4. Features: TF-IDF on word 1-2-grams  +  TF-IDF on character 3-5-grams
  5. Compare 3 classic classifiers with 5-fold stratified cross-validation (F1 on spam):
        - Multinomial Naive Bayes
        - Logistic Regression (class-balanced)
        - Linear SVM (class-balanced, probability-calibrated)
  6. Refit the winner on the full training set, evaluate once on the untouched test set
  7. Save model + metrics (accuracy / precision / recall / F1 / confusion matrix /
     model comparison / most spam- and ham-indicative words) for the GUI

Run:  python train_model.py            (uses data/spam.csv)
      python train_model.py --csv path/to/other.csv
"""
from __future__ import annotations

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from spam_engine import DATA_PATH, METRICS_PATH, MODEL_PATH, MODELS_DIR, normalize_text

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
def load_dataset(csv_path: str) -> tuple[pd.DataFrame, dict]:
    """Read the CSV robustly (the classic file is cp1252, not UTF-8)."""
    last_err = None
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            raw = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    else:
        raise ValueError(f"Could not read {csv_path}: {last_err}")

    cols = {c.lower().strip(): c for c in raw.columns}
    label_col = next((cols[k] for k in ("v1", "label", "class", "category", "type") if k in cols), raw.columns[0])
    text_col = next((cols[k] for k in ("v2", "text", "message", "sms", "email", "body") if k in cols), raw.columns[1])

    df = raw[[label_col, text_col]].copy()
    df.columns = ["label", "text"]
    df = df.dropna()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham", "1", "0"])]
    df["y"] = df["label"].isin(["spam", "1"]).astype(int)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    n_raw = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    info = {
        "rows_raw": int(n_raw),
        "rows_clean": int(len(df)),
        "duplicates_removed": int(n_raw - len(df)),
        "n_spam": int(df["y"].sum()),
        "n_ham": int((1 - df["y"]).sum()),
        "avg_len_spam": float(df.loc[df.y == 1, "text"].str.len().mean()),
        "avg_len_ham": float(df.loc[df.y == 0, "text"].str.len().mean()),
    }
    if info["n_spam"] < 10 or info["n_ham"] < 10:
        raise ValueError("Dataset needs at least 10 spam and 10 ham messages.")
    return df, info


def build_features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2),
                                 min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=normalize_text, analyzer="char_wb",
                                 ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])


def candidate_models() -> dict:
    return {
        "Naive Bayes": MultinomialNB(alpha=0.05),
        "Logistic Regression": LogisticRegression(C=15, class_weight="balanced",
                                                  max_iter=3000, solver="liblinear"),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(C=0.6, class_weight="balanced"), cv=5),
    }


def _scores(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def top_words(train_text, y_train, n: int = 12) -> tuple[list, list]:
    """Most spam- / ham-indicative words (from a simple word-level logistic model)."""
    # "lt"/"gt"/"amp" are HTML-escape leftovers in this dataset, not real words.
    stop = list(ENGLISH_STOP_WORDS | {"lt", "gt", "amp", "ll", "ve"})
    vec = TfidfVectorizer(preprocessor=normalize_text, min_df=3, stop_words=stop, sublinear_tf=True)
    X = vec.fit_transform(train_text)
    lr = LogisticRegression(C=10, class_weight="balanced", max_iter=2000, solver="liblinear").fit(X, y_train)
    friendly = {"longnumtoken": "[long number]", "urltoken": "[web link]", "currencytoken": "[£ / $ symbol]"}
    names = np.array([friendly.get(n, n) for n in vec.get_feature_names_out()])
    order = np.argsort(lr.coef_[0])
    spam = [(str(names[i]), float(lr.coef_[0][i])) for i in order[::-1][:n]]
    ham = [(str(names[i]), float(lr.coef_[0][i])) for i in order[:n]]
    return spam, ham


# --------------------------------------------------------------------------- #
def train(csv_path: str = DATA_PATH, log=print) -> dict:
    t0 = time.time()
    log(f"Loading {csv_path} ...")
    df, info = load_dataset(csv_path)
    log(f"  {info['rows_clean']} unique messages "
        f"({info['n_spam']} spam / {info['n_ham']} ham, {info['duplicates_removed']} duplicates removed)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["y"], test_size=0.20, stratify=df["y"], random_state=RANDOM_STATE)
    log(f"  train={len(X_train)}  test={len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison, fitted = [], {}
    for name, clf in candidate_models().items():
        pipe = Pipeline([("features", build_features()), ("clf", clf)])
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        sc = _scores(y_test, (prob >= 0.5).astype(int), prob)
        sc.update({"model": name, "cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std()),
                   "cv_auc_mean": float(cv_auc.mean())})
        comparison.append(sc)
        fitted[name] = pipe
        log(f"  {name:<20} CV F1 {cv_f1.mean():.4f} (+/-{cv_f1.std():.4f})  CV AUC {cv_auc.mean():.4f} | "
            f"test acc {sc['accuracy']:.4f}  P {sc['precision']:.4f}  R {sc['recall']:.4f}  F1 {sc['f1']:.4f}")

    # Selection uses CROSS-VALIDATION ONLY (the test set stays untouched).
    # Differences in CV F1 below ~0.005 are inside the fold-to-fold noise, so among
    # all models within that band we prefer the best ranking quality (CV ROC-AUC),
    # which is what makes the app's sensitivity slider behave well.
    top_f1 = max(c["cv_f1_mean"] for c in comparison)
    contenders = [c for c in comparison if c["cv_f1_mean"] >= top_f1 - 0.005]
    best_name = max(contenders, key=lambda c: c["cv_auc_mean"])["model"]
    best_pipe = fitted[best_name]

    prob = best_pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    final = _scores(y_test, pred, prob)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    spam_w, ham_w = top_words(X_train, y_train)

    metrics = {
        "best_model": best_name,
        "dataset": info,
        "test_size": int(len(X_test)),
        "test": final,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "comparison": comparison,
        "top_spam_words": spam_w,
        "top_ham_words": ham_w,
        "trained_on": os.path.basename(csv_path),
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "train_seconds": round(time.time() - t0, 1),
    }

    # Ship a model fitted on ALL data for deployment (more data = slightly better),
    # but the metrics above stay honest because they come from the held-out split.
    best_pipe.fit(df["text"], df["y"])

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH, compress=3)
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    log(f"\nBest model: {best_name}")
    log(f"Held-out test -> acc {final['accuracy']:.4f} | precision {final['precision']:.4f} | "
        f"recall {final['recall']:.4f} | F1 {final['f1']:.4f} | AUC {final['roc_auc']:.4f}")
    log(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    log(f"Saved {MODEL_PATH}  ({metrics['train_seconds']}s)")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the spam classifier")
    ap.add_argument("--csv", default=DATA_PATH, help="path to labelled CSV (default: data/spam.csv)")
    train(ap.parse_args().csv)
