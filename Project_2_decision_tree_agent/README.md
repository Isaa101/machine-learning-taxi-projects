# Project 2: Rule-Based Agent vs Decision Tree Agent

This project compares a heuristic-based agent with a machine learning agent trained using a Decision Tree classifier. Both agents operate in the Taxi-v3 environment using the dataset generated in Tutorial 1.

---
## Contents

- `Non_ML_Agent.py`: Implements a rule-based agent using Manhattan distance and short-term memory.
- `First_Decision_Tree.py`: Trains a DecisionTreeClassifier using the logged experiences.
- `tune_dt.py`: Performs parameter tuning on the decision tree model.
- `dt_agent.py`: Deploys the trained decision tree model in the Taxi-v3 environment.

## Features

- Heuristic agent with memory to avoid loops and improve navigation.
- Decision tree trained on spatial features (taxi row, col, passenger location, destination).
- Evaluation metrics: accuracy, precision, recall, F1-score.
- Confusion matrix and tree visualization for interpretability.

## How to Run

1. Install dependencies:
   ```bash
   pip install scikit-learn gymnasium matplotlib

2. Train the decision tree:
   ```bash
   python First_Decision_Tree.py

3. Tune parameters:
   ```bash
   python tune_dt.py

4. Deploy the agent:
   ```bash
   python dt_agent.py

## Results

- Default model accuracy: 95.6%
- Tuned model accuracy: 95.59%
- Most errors occur in adjacent movement predictions (e.g., Up vs. Left).
- Pickup and Dropoff actions are predicted correctly.

## Conclusion
The rule-based agent is simple and explainable but limited in adaptability. The decision tree agent generalizes better and performs consistently across episodes, making it a scalable solution for more complex environments.

---
