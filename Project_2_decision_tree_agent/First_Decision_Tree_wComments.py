#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tutorial 2 - Decision Trees
First Decission Tree

Authors: Carlota Campos Rubio, Isabel Gregorio Diez

Train & evaluate a Decision Tree classifier to predict Taxi-v3 actions from tabular experiences.

"""

# --------------------------- Standard library -----------------------------
from pathlib import Path  # Path provides convenient, cross-platform file/folder path handling.

# --------------------------- Third-party libs -----------------------------
import joblib              # For persisting (dump/load) trained models to disk in a compact format.
import pandas as pd        # For loading CSV files and manipulating tabular data (DataFrames).
import matplotlib.pyplot as plt  # For plotting and saving the decision tree visualization.

# Scikit-learn utilities:
from sklearn.model_selection import train_test_split  # Splits dataset into train/test subsets.
from sklearn.tree import (
    DecisionTreeClassifier,  # The decision tree classifier algorithm (CART).
    plot_tree,               # Utility to visualize the trained tree with matplotlib.
    export_text              # Utility to export the tree rules as plain text.
)
from sklearn.metrics import (
    accuracy_score,                    # Overall fraction of correct predictions.
    precision_recall_fscore_support,   # Precision/Recall/F1 (supports macro/weighted/micro averaging).
    classification_report,             # Comprehensive per-class report as a string.
    confusion_matrix                   # Confusion matrix (rows: true labels, cols: predicted labels).
)

# ------------------------------------------------------------
# Config (file paths, feature names, label name, class names)
# ------------------------------------------------------------

# Path to the CSV dataset of experiences.
DATA = Path("experiences.csv") #In the same folder as the script

# Output folder for model and artifacts (created if missing).
OUT = Path("models"); OUT.mkdir(exist_ok=True)

# Feature columns expected in the CSV (order matters for readability in exports).
FEATURES = ["taxi_row", "taxi_col", "pass_loc", "dest_idx"]

# Label column name: the action taken by the agent (integer class 0..5).
LABEL = "action"

# Human-readable class names aligned with the label indices (0..5).
ACTION_NAMES = ["Down", "Up", "Right", "Left", "Pickup", "Dropoff"]



# ------------------------------------------------------------
# Training & evaluation (defaults)
# ------------------------------------------------------------

def main():
    # ------------------
    # Load the dataset
    # ------------------
    # Read the CSV into a DataFrame (df). Pandas infers dtypes automatically.
    df = pd.read_csv(DATA)

    # --------------------------
    # Prepare features and label
    # --------------------------
    # X: feature matrix (state)
    X = df[FEATURES].copy()  # copy() to avoid chained assignment surprises downstream

    # y: target vector (action index in {0..5})
    y = df[LABEL].astype(int) # Ensure integer type for classification.

    # ----------------------------------------------
    # Train/test split (80/20), stratified by class
    # ----------------------------------------------
    # - test_size=0.20 => 20% test set, 80% training set (as required).
    # - random_state=42 => deterministic split for reproducibility across runs.
    # - stratify=y => preserves the label distribution proportionally in train and test.
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )  

    # --------------------------------------
    # Model: Decision Tree (default params)
    # --------------------------------------
    # We fix random_state=42 for deterministic behavior (ties, feature ordering, etc.).
    model = DecisionTreeClassifier(random_state=42)
    # Fit/train the model on the training partition.
    model.fit(Xtr, ytr)

    # ----------------------
    # Predictions & metrics
    # ----------------------
    # Predict labels for the held-out test set.
    ypred = model.predict(Xte)

    # Accuracy: fraction of correct predictions over total.
    acc = accuracy_score(yte, ypred)

    # Macro-averaged Precision, Recall, F1:
    # - Computes the metric independently for each class, then takes the unweighted mean.
    # - zero_division=0 avoids warnings and sets metric to 0 when a class has no predicted samples.
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        yte, ypred, average="macro", zero_division=0
    )

    # -------------------------
    # Correct / incorrect counts
    # -------------------------
    # Count how many predictions match the ground truth.
    correct = int((ypred == yte).sum())

    # Total number of examples in the test set (length of yte).
    total = yte.shape[0]

    # Incorrect predictions are the remainder.
    incorrect = total - correct

    # -------------------------
    # Print evaluation summary
    # -------------------------
    print(f"\n=== Decision Tree (defaults) ===")
    print(f"Correct:     {correct}")
    print(f"Incorrect:   {incorrect}")
    print(f"Accuracy:    {acc:.3f}")
    print(f"Precision (macro): {prec_macro:.3f}")
    print(f"Recall (macro):    {rec_macro:.3f}")
    print(f"F1 (macro):        {f1_macro:.3f}\n")

    # Per-class classification report:
    # Includes precision, recall, F1, and support for each class, plus macro/weighted averages.
    # `target_names` maps the numeric classes (0..5) to human-readable labels for readability.
    print("Per-class classification report (0..5):")
    print(classification_report(yte, ypred, target_names=ACTION_NAMES, zero_division=0))

    # Confusion matrix:
    # - Rows represent the true classes.
    # - Columns represent the predicted classes.
    # Diagonal entries are correct predictions; off-diagonals indicate confusions.
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(yte, ypred))

    # -------------------------
    # Persist artifacts to disk
    # -------------------------

    # 1) Save the trained model to 'models/dt_default.pkl'.
    # Use joblib for efficiency with scikit-learn estimators.
    joblib.dump(model, OUT / "dt_default.pkl")

    # 2) Export the tree as human-readable text rules (one rule path per leaf).
    # This is very handy for debugging and explaining decisions.
    rules_txt = export_text(model, feature_names=FEATURES)
    (OUT / "tree_default.txt").write_text(rules_txt, encoding="utf-8")

    # 3) Export a PNG image of the tree (useful for reports).
    # - figsize controls the canvas size (in inches).
    # - filled=True colors nodes by predicted class for readability.
    # - class_names shows the human-readable labels on leaves.
    plt.figure(figsize=(14, 9))
    plot_tree(model, feature_names=FEATURES, class_names=ACTION_NAMES, filled=True)
    plt.tight_layout()  # Reduce overlapping labels/margins.
    plt.savefig(OUT / "tree_default.png", dpi=300) 
    plt.close()         # Free the figure to avoid memory leaks in repeated runs.

    # -------------------------
    # Summary of saved outputs
    # -------------------------
    print("\nFiles saved under 'models/':")
    print(" - dt_default.pkl   (trained model)")
    print(" - tree_default.txt (textual rules)")
    print(" - tree_default.png (tree image)")


if __name__ == "__main__":
    main()