#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading 
import queue
import time

import gymnasium as gym # To import and control the taxi env
from pynput import keyboard # To read the keyboard
from map_loader import prepare_for_env # Converts the text map

# We declare which key corresponds to each action
KEY_TO_ACTION = {
    'down': 0,   # South
    'up': 1,     # North
    'right': 2,  # East
    'left': 3,   # West
    'space': 4,  # Pickup
    'enter': 5,  # Dropoff
}

class KeyboardController:
    def __init__(self):
        self.actions = queue.Queue() # Queue where we store the actions
        self.reset_requested = False # Set to false until we want to reset and press r
        self.quit_requested = False # Set to false until we want to quit and press q
        self.listener = keyboard.Listener(on_press=self.on_press) # We create a listener that reacts each time we press a key, it calls self.on_press()
        self.listener.daemon = True # If the program finishes, we don't need to wait for the thread to finish

    def start(self):
        self.listener.start()

    def on_press(self, key):
        try:
            if key == keyboard.Key.up:
                self.actions.put(KEY_TO_ACTION['up'])
            elif key == keyboard.Key.down:
                self.actions.put(KEY_TO_ACTION['down'])
            elif key == keyboard.Key.left:
                self.actions.put(KEY_TO_ACTION['left'])
            elif key == keyboard.Key.right:
                self.actions.put(KEY_TO_ACTION['right'])
            elif key == keyboard.Key.space:
                self.actions.put(KEY_TO_ACTION['space'])
            elif key == keyboard.Key.enter:
                self.actions.put(KEY_TO_ACTION['enter'])

            elif hasattr(key, 'char') and key.char is not None: # We check that the entered key is a letter and not another symbol 
                c = key.char.lower()
                if c == 'r':
                    self.reset_requested = True
                elif c == 'q':
                    self.quit_requested = True
        except Exception:
            # Avoid an exception stops a thread 
            pass

def main():
    # Import the map
    desc = prepare_for_env('map_1.txt')
    # Create the env
    env = gym.make("Taxi-v3", desc=desc, render_mode="human")
    # Obs is the initial state and we use info to give extra information
    # Note: if we set a seed, the initial state will always be the same; if we don't, it will be random
    obs, info = env.reset(seed=42) 

    ctrl = KeyboardController()
    ctrl.start()

    episode = 1
    total_reward = 0.0
    print(f"Episode {episode} — use arrow/space/enter; 'r' for reset, 'q' for exit")

    try:
        while True:
            # Exit
            if ctrl.quit_requested:
                print("Exit because of 'q'...")
                break

            # Reset
            if ctrl.reset_requested:
                obs, info = env.reset()
                total_reward = 0.0
                episode += 1
                ctrl.reset_requested = False
                print(f"Episode {episode} — reset")

            # Wait for the user to enter an action
            try:
                action = ctrl.actions.get(timeout=0.05) 
            except queue.Empty:
            # To show the environment even if the user does not press any key
                env.render() 
                continue

            # Check action_mask to see if an action is possible or not 
            mask = info.get("action_mask", None)
            if mask is not None and mask[action] == 0:
                #Not valid action, but we are going to take it into account so we can see the penalty 
                pass
            
            # Get the new state, reward, and whether the episode ended (by success or by timeout)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(f"Episode {episode} finished — total reward: {total_reward:.1f}")
                obs, info = env.reset()
                total_reward = 0.0
                episode += 1
                print(f"Episode {episode} — ready")

            #We need to perform a little pause so it does not go very fast 
            time.sleep(0.02)

    finally:
        env.close()

if __name__ == "__main__":
    main()
