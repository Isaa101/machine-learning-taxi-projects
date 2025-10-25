# Machine Learning Projects in Taxi-v3 Environment

This repository contains a series of practical experiments in machine learning and behavioral cloning using the Taxi-v3 environment from Gymnasium. The work focuses on manual control, rule-based heuristics, and supervised learning using decision trees.

## Overview

The goal is to understand and implement different approaches to agent control, starting from human demonstrations and progressing to machine learning models. The environment simulates a taxi navigating a grid to pick up and drop off passengers, providing a rich setting for reinforcement learning and behavioral cloning.

## Structure

### `Project_1_keyboard_agent`
- Implements a keyboard-controlled agent using `pynput`.
- Logs state-action pairs to generate a dataset for behavioral cloning.
- Includes analysis of action distribution and state visitation.

### `Project_2_decision_tree_agent`
- Compares a rule-based agent with a DecisionTreeClassifier.
- Trains and deploys a machine learning model using the dataset from Tutorial 1.
- Includes parameter tuning and performance evaluation.

## Technologies Used

- Python
- Gymnasium
- Scikit-learn
- Pynput
- Matplotlib
- CSV data analysis

## How to Use

Each tutorial folder includes its own `README.md` with instructions on how to run the code, dependencies, and expected results.

## Authors

This repository was developed by:

- **Isabel Gregorio Díez**  
- **Carlota Campos Rubio**

As part of a hands-on exploration in machine learning for robotics, using the Taxi-v3 environment from Gymnasium. The work reflects personal learning progress and experimentation with agent design, behavioral cloning, and decision tree models.

## License
This project is open-source and available under the MIT License.
