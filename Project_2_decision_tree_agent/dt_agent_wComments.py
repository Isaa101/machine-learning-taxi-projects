#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import gymnasium as gym

#From practice 1
from map_loader import prepare_for_env 

FEATURES = ["taxi_row", "taxi_col", "pass_loc", "dest_idx"]
ACTION_NAMES = ["Down","Up","Right","Left","Pickup","Dropoff"]

#Helpers of state and actions
def decode_state(env, obs):
    # Taxi-v3 codifies the state as an int. Therefore, env.unwrapped.decode it descompose it as a tuple
    return env.unwrapped.decode(obs)

def rule_fallback(env, obs, info):
    """Plan B: if the model gives invalid actions, choose a valid action"""
    r, c, pass_loc, dest_idx = decode_state(env, obs) #Decomposed state
    locs = env.unwrapped.locs #coordenates of the 4 stops
    mask = info.get("action_mask") #To know if an action is valid/legal (1) or invalid/ilegal (0)

    in_taxi = (pass_loc == len(locs)) #True if the passenger is already inside the taxi

    #If I can do Dropoff/Pickup and is a legal action, it has priority
    if in_taxi and mask[5] == 1:
        return 5
    if not in_taxi and mask[4] == 1:
        return 4

    # If I can not do a Dropoff/Pickup, we try to move to the target (passenger/destination)
    tr, tc = (locs[dest_idx] if in_taxi else locs[pass_loc])
    if tr > r and mask[0] == 1: return 0  # Down
    if tr < r and mask[1] == 1: return 1  # Up
    if tc > c and mask[2] == 1: return 2  # Right
    if tc < c and mask[3] == 1: return 3  # Left

    #If any of the actions above are possible, returns the first legal action
    for a in range(6):
        if mask[a] == 1:
            return a
    return 0  

def choose_with_mask(model, X_row, mask, classes_):
    """Returns the best action **legal** depending on the model"""
    #Direct prediction
    pred = int(model.predict([X_row])[0])
    if mask[pred] == 1:
        return pred, False  # False = is valid, use it

    # If the direct is not valid, use probability to decide which is the most probable valid action
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([X_row])[0]  # probability for each class
        scores = np.full(6, -1.0) #creates an NumPy array of lenght 6 with -1 initial value to ignore the actions without probability 
        for i, cls in enumerate(classes_):
            scores[int(cls)] = proba[i]
        #We choose the best valid action (with mask=1)
        best_valid = None
        best_score = -1.0
        for a in range(6):
            if mask[a] == 1 and scores[a] > best_score:
                best_valid = a
                best_score = scores[a]
        if best_valid is not None:
            return int(best_valid), True  # True = the direct was invalid

    #If everything in the model is invalid, we return None so that the caller makes fallback
    return None, True

#Evaluation loop

def run_agent(model_path, episodes=10, max_steps=200, render=False, sleep=0.02, seed=42):
    #Prepare the environment (the map)
    desc = prepare_for_env('map_1.txt')
    render_mode = "human" if render else None
    env = gym.make("Taxi-v3", desc=desc, render_mode=render_mode)

    #Put the model and the classes 
    model = joblib.load(model_path)
    classes_ = model.classes_

    results = [] #foe the tuples in each episode
    invalid_preds = 0 #Num of times the direct prediction was invalid according to the mask
    used_fallbacks = 0 #Num of times we had to do rule_fallback

    try:
        for ep in range(1, episodes + 1):
            obs, info = env.reset(seed=seed + ep)  #New seed in each episode
            steps = 0
            ep_reward = 0.0

            for t in range(max_steps):
                r, c, pass_loc, dest_idx = decode_state(env, obs)
                X_row = [r, c, pass_loc, dest_idx]
                mask = info.get("action_mask")

                action, was_invalid = choose_with_mask(model, X_row, mask, classes_)
                if action is None:
                    action = rule_fallback(env, obs, info) #apply plan B according to the rules
                    used_fallbacks += 1
                if was_invalid: #how many times the direct was invalid
                    invalid_preds += 1

                #Executes the action in the environment 
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                steps += 1

                if render:
                    time.sleep(sleep) #just a little pause for the visual 

                if terminated or truncated:
                    success = bool(terminated)  # terminates = True means that the passenger reached the destination
                    results.append((ep_reward, steps, success)) #Save the statistics of the episode
                    print(f"[Ep {ep:02d}] reward={ep_reward:.1f}  steps={steps}  success={success}")
                    break
            else:
                #If there was not a break (it did not reach the goal during max_steps)
                results.append((ep_reward, steps, False))
                print(f"[Ep {ep:02d}] reached max_steps -> success=False")

    finally:
        env.close()

    #Summary of the episodes
    rewards = [r for (r, s, ok) in results]
    stepss = [s for (r, s, ok) in results]
    oks = [ok for (r, s, ok) in results]
    summary = {
        "episodes": episodes,
        "success_rate": float(np.mean(oks)),
        "avg_reward": float(np.mean(rewards)),
        "avg_steps": float(np.mean(stepss)),
        "invalid_pred_actions": invalid_preds,
        "fallback_uses": used_fallbacks,
    }
    return summary

def parse_args():
    #To define the possible key arguments and their values by default 
    p = argparse.ArgumentParser(description="Run DecisionTree Taxi agent")
    p.add_argument("--model", type=Path, default=Path("models/dt_best.pkl"), help="Rute to the .pkl  of the model (default: models/dt_best.pkl)")
    p.add_argument("--episodes", type=int, default=10, help="Num of episodes")
    p.add_argument("--max-steps", type=int, default=200, help="Max of steps in the episode")
    p.add_argument("--render", action="store_true", help="Draw the environment")
    p.add_argument("--sleep", type=float, default=0.02, help="Pause between steps when --render")
    return p.parse_args()

def main():
    args = parse_args()
    summary = run_agent(
        model_path=args.model,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        sleep=args.sleep,
    )
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
