#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import itertools
import joblib
import pandas as pd
import matplotlib.pyplot as plt

#To separate train and test
from sklearn.model_selection import train_test_split 
#Plot_tree is the function to draw the tree
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DATA = Path("experiences.csv")
#Exit folder (if it does not exist, we create it)
OUT = Path("models"); OUT.mkdir(exist_ok=True)
FEATURES = ["taxi_row", "taxi_col", "pass_loc", "dest_idx"]
LABEL = "action"
ACTION_NAMES = ["Down","Up","Right","Left","Pickup","Dropoff"]

#Values we will test. Total num of combinations: 3x2x5x3x2 = 180
GRID = {
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": ["best", "random"],
    "max_depth": [None, 4, 6, 8, 12],
    "min_samples_split": [2, 5, 10],
    "class_weight": [None, "balanced"],
}

#It makes all the possible combinations 
def iter_configs(grid):
    #We have here a list of all the parameters in in a stable order. Example: ('gini','best',None,2,None)
    keys = list(grid.keys())
    # '*' unpacks the lists as separate arguments to make the castesian product()
    for values in itertools.product(*(grid[k] for k in keys)): 
        #Gives the dictionary but thanks to yield, gives one at a time
        yield dict(zip(keys, values)) 

def main():
    df = pd.read_csv(DATA)
    X = df[FEATURES].copy()
    y = df[LABEL].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    #Acumulates the results of all the combinations (for the table)
    rows = [] 
    #Later when we have the best tree, we will have: acc (accuracy), f1 (F1 macro), cfg (hiperparameters of that combination), model(trained tree)
    best = None  

    for cfg in iter_configs(GRID):
        #** used to pass all the hiperparameters 
        model = DecisionTreeClassifier(random_state=42, **cfg)
        model.fit(Xtr, ytr)

        #Predicts the test
        ypred = model.predict(Xte)
        #To calculate accuracy, precision, recall, F1 macro
        acc = accuracy_score(yte, ypred) 
        prec, rec, f1, _ = precision_recall_fscore_support(
            yte, ypred, average="macro", zero_division=0 #zero_division= 0 avoids if any class does not apear 
        )

        rows.append({**cfg,
                     "accuracy": acc,
                     "precision_macro": prec,
                     "recall_macro": rec,
                     "f1_macro": f1})

        #Compares and gets the best model. If there is a draw, F1 decides
        if (best is None) or (acc > best[0]) or (acc == best[0] and f1 > best[1]):
            best = (acc, f1, cfg, model)

    #Creates the table with all the combinations, sort and store it at the CSV
    table = pd.DataFrame(rows).sort_values(
        ["accuracy","f1_macro","precision_macro"], ascending=False
    )
    table.to_csv(OUT / "dt_grid_results.csv", index=False)
    print("Top 10 combinations:")
    print(table.head(10).to_string(index=False))

    # Stores the best model
    best_acc, best_f1, best_cfg, best_model = best
    # Save (serialize) the best trained decision tree to disk at models/dt_best.pkl
    joblib.dump(best_model, OUT / "dt_best.pkl")
    (OUT / "dt_best_config.txt").write_text(str(best_cfg), encoding="utf-8")
    print("\nBest configuration:")
    print(best_cfg)
    print(f"Accuracy: {best_acc:.3f} | F1 macro: {best_f1:.3f}")

    # Image of the best tree
    plt.figure(figsize=(18, 12)) #18 and 12 stes the figure size 
    plot_tree(best_model, feature_names=FEATURES, class_names=ACTION_NAMES,
              filled=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "tree_best.png", dpi=220, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
