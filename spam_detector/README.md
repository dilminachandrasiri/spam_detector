# 🛡 SpamShield — AI Spam Detector

A desktop app with a modern dark/light GUI that classifies messages as **spam** or **ham** (not spam) using a machine-learning model trained on the SMS Spam Collection dataset (`data/spam.csv`, 5,572 messages).

## Quick start (Windows)

**1. Download the project**
Click the green **Code** button → **Download ZIP**, then right-click the zip and choose **Extract All**.

**2. Open a terminal in the folder that contains `app.py`**
The zip extracts to a folder inside a folder, so you may need to go one level deeper:

```powershell
cd spam_detector-main
cd spam_detector
```

Check you're in the right place by running `dir`. You should see `app.py` and `requirements.txt` in the list.

**3. (Recommended) Create a virtual environment**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**4. Install the libraries**

```powershell
python -m pip install -r requirements.txt
```

**5. Run the app**

```powershell
python app.py
```

The first launch trains the model automatically (about 20 seconds) and saves it to `models/`.

> **Mac / Linux:** use `python3` instead of `python`, and `source .venv/bin/activate` to activate the environment.
> Requires Python 3.9+ with Tk. It is included in the python.org installers for Windows and macOS. On Ubuntu/Debian: `sudo apt install python3-tk`

## Troubleshooting

| Error | Cause and fix |
|---|---|
| `Could not open requirements file: No such file or directory: 'requirements.txt'` | You are in the wrong folder. Run `cd spam_detector` (or whichever folder holds `app.py`), then run the install command again. |
| `No module named 'pandas'` (or `customtkinter`, `sklearn`) | The libraries are not installed in the Python you are running. Run `python -m pip install -r requirements.txt`. Always use `python -m pip`, not plain `pip`, so the libraries go into the same Python that runs the app. |
| "Training failed: No module named ..." in the app window | Same cause as above. Install the requirements, then restart the app. |
| `No module named 'tkinter'` | Python was installed without Tk. Reinstall Python from python.org and tick **tcl/tk and IDLE**. |
| `python` is not recognized | Install Python from python.org and tick **Add python.exe to PATH**, then reopen the terminal. |
| Using PyCharm | Open the folder that contains `app.py` as the project. The Terminal tab should show `(.venv)` at the start of the line before you install. |

## Features

| Page | What it does |
|---|---|
| **Scan message** | Paste text and get a verdict, spam probability, the words that pushed the score, red flags and a sensitivity slider. Press `Ctrl+Enter` to analyze. |
| **Batch scan** | Open a CSV or a `.txt` file (one message per line), see the spam rate, filter results and export them to CSV. |
| **Model insights** | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, model comparison, top spam words, and a Retrain button. |
| **History** | Everything you checked in this session. |

## The ML pipeline

1. Load the CSV and remove duplicate messages (prevents train/test leakage).
2. Stratified 80/20 train/test split.
3. Normalize text: links, long numbers and currency symbols become generic tokens.
4. Features: TF-IDF on word 1–2-grams plus character 3–5-grams.
5. Compare Naive Bayes, Logistic Regression and a calibrated Linear SVM with 5-fold cross-validation.
6. Evaluate the winner once on the untouched test set.
7. Explain predictions by removing each word and measuring the change in the model's score.

**Held-out test results:** ~99.3% accuracy, ~97.6% precision, ~96.9% recall, ROC-AUC ~0.999.

## Limitations

- The dataset is English SMS from around 2005, mostly UK, so modern phishing and other languages need retraining on matching data.
- The shipped model has seen `data/spam.csv`, so scanning that file in Batch scan is not a fair test.

## Project files

```
app.py            GUI (CustomTkinter)
spam_engine.py    prediction, explanations, text cleaning
train_model.py    training and evaluation pipeline
data/spam.csv     dataset
requirements.txt  dependencies
```

## Tech stack

Python · scikit-learn · pandas · CustomTkinter
