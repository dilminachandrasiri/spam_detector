# spam_detector
AI-powered spam detector with a modern desktop GUI. Classifies SMS/messages as spam or ham using TF-IDF + machine learning (Naive Bayes, Logistic Regression, Linear SVM) with word-level explanations. Built with Python, scikit-learn and CustomTkinter.


# SpamShield — AI Spam Detector

A desktop app (modern dark/light GUI) that classifies messages as **spam** or **ham** using a
machine-learning model trained on the SMS Spam Collection (`data/spam.csv`, 5,572 messages).

## Run it

```bash
pip install -r requirements.txt
python app.py
```

First launch trains the model automatically (~20 s) and saves it to `models/`.
You can also train from the terminal: `python train_model.py` (or `--csv other.csv`).

> Needs Python 3.9+ with Tk (included in the python.org installers for Windows/macOS;
> on Debian/Ubuntu: `sudo apt install python3-tk`).

## Features
| Page | What it does |
|---|---|
| **Scan message** | Paste text → verdict, spam probability, **which words pushed the score**, rule-based red flags, sensitivity slider, spam words highlighted in your text. `Ctrl+Enter` to analyze. |
| **Batch scan** | Open a CSV (auto-detects the text column) or a `.txt` (one message per line), see spam rate, filter All/Spam/Ham, export results to CSV. |
| **Model insights** | Accuracy / precision / recall / F1 / ROC-AUC, confusion matrix, 3-model comparison, top spam & ham words, **Retrain** button, train on your own labelled CSV. |
| **History** | Everything checked this session; double-click to re-open. |

## The ML pipeline (`train_model.py`)
1. Load CSV (cp1252-safe), drop junk columns, **remove 414 duplicate messages** (prevents train/test leakage).
2. Stratified 80/20 split (spam is only ~12 % of the data).
3. Text normalisation: URLs / long numbers / currency symbols → generic tokens; `å£` encoding bug repaired.
4. Features: TF-IDF **word 1–2-grams + character 3–5-grams**.
5. Candidates: Multinomial Naive Bayes, Logistic Regression, Calibrated Linear SVM — compared with
   **5-fold stratified CV** (F1 on the spam class); ties inside the noise band broken by CV ROC-AUC.
6. Winner evaluated **once** on the untouched test set → numbers shown in the app are honest.
   The shipped model is then refit on all data.
7. Explanations: leave-one-word-out in log-odds space (works with any classifier).

Typical held-out result: **~99.3 % accuracy, ~97.6 % precision, ~96.9 % recall, ROC-AUC ~0.999.**

## Notes
* Scanning `data/spam.csv` itself in Batch mode isn't a fair test — the shipped model has seen it.
  Use fresh messages, or trust the held-out numbers in *Model insights*.
* The dataset is English SMS from ~2005 (UK-heavy). Modern phishing, other languages or email
  will need retraining on matching data — use *Train on another CSV…* (needs a label column
  with `spam`/`ham` and a text column).

## Files
```
app.py           GUI (customtkinter)
spam_engine.py   inference, explanations, text normalisation
train_model.py   training / evaluation pipeline
data/spam.csv    dataset
models/          saved model + metrics.json (created on first run)
```
