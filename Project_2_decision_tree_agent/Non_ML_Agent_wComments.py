#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tutorial 2 - Decision Trees
Non ML Agent

Authors: Carlota Campos Rubio, Isabel Gregorio Diez

Core idea:
- The environment is the classic Taxi problem: move a taxi on a grid, pick up a passenger, and drop them off.
- This agent is not learning; it uses heuristics + a short-term "tabu-like" memory:
  - Avoid recently visited positions within the same phase (pickup or dropoff).
  - Limit how many times we step on the same cell during a phase to reduce oscillations.
  - Move greedily (by Manhattan distance) toward the current target (passenger location or destination).
- The memory is reset when we switch from the pickup phase to the dropoff phase, and at episode reset.

"""

# --- Standard library imports --------------------------------------------
import time  # Used to throttle the main loop slightly for human-friendly rendering/speed.

# Collections utilities:
#   - deque: a fast, bounded queue we use to store recent positions (tabu list).
#   - defaultdict: dict that auto-initializes missing keys (handy for visit counters).
from collections import deque, defaultdict

# --- Third-party libraries -----------------------------------------------
# Gymnasium is a maintained fork of OpenAI Gym. Used to create and step through Taxi-v3.
import gymnasium as gym

# --- Project/local utilities ---------------------------------------------
# Custom map description compatible with Taxi-v3.
from map_loader import prepare_for_env  

# -------------------------------------------------------------------------
# ACTION SPACE 
#   0: South, 1: North, 2: East, 3: West, 4: Pickup, 5: Dropoff
ACTIONS = ["South", "North", "East", "West", "Pickup", "Dropoff"]

# ==== SHORT-TERM MEMORY ===================================================
# These knobs control how "tabu" and "visit limits" behave within a phase.
TABU_LEN = 8               # How many recent positions we try to avoid (per phase).
MAX_VISITS_PER_PHASE = 2   # Max times we allow stepping onto the same cell in the same phase.

# In-episode memory structures (RESET every episode and when switching to dropoff phase):
LAST_POSITIONS = deque(maxlen=TABU_LEN)   # stores recent positions as (phase, (row, col)).
VISITS = defaultdict(int)                 # counts how many times we've visited a cell in the current phase. Key: (phase, row, col) -> count.
LAST_PHASE = None                         # tracks current phase string: either "pickup" or "dropoff".
# ==========================================================================


# ============================== Helpers ===================================

def decode_state(env, obs):
    #Convert the integer observation `obs` from Taxi-v3 into a human-readable tuple.

    taxi_row, taxi_col, pass_loc, dest_idx = env.unwrapped.decode(obs)
    return taxi_row, taxi_col, pass_loc, dest_idx


def phase_from(pass_loc, locs):
    #Determine phase from passenger location index.
    
    return "dropoff" if pass_loc == len(locs) else "pickup"


def next_pos(r, c, a):
    #Compute the next (row, col) position if the taxi were to take action `a` (0..5).
    
    if a == 0:   return (r + 1, c)  # South
    if a == 1:   return (r - 1, c)  # North
    if a == 2:   return (r, c + 1)  # East
    if a == 3:   return (r, c - 1)  # West
    return (r, c)                   # Pickup/Dropoff and anything else fall back to no move


def manhattan(p, q):
    """
    Manhattan (L1) distance between two grid points p=(r1,c1) and q=(r2,c2).
    This is appropriate for 4-neighborhood grid movement.
    """
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def move_towards(src, dst, mask, phase):
    """
    Greedy movement toward the target `dst` while respecting:
      - `mask` (action mask from the environment, typically info["action_mask"])
      - short-term memory (tabu positions and visit counts in the current phase)

    Preference order (lower "cost" is better):
      1) Avoid recently visited (tabu) positions in the same phase if there are alternatives.
      2) Avoid positions that already reached MAX_VISITS_PER_PHASE in this phase.
      3) Prefer fewer visits.
      4) Prefer smaller Manhattan distance to the target.
      5) Tie-breaker by our heuristic candidate order.

    """
    # If no mask is provided, assume all actions are allowed for safety.
    if mask is None:
        mask = [1, 1, 1, 1, 1, 1]

    (r, c), (tr, tc) = src, dst
    candidates = []  # candidate movement actions in heuristic order

    # Heuristic priority by the axis with larger absolute distance.
    # We try to reduce the dominant axis first (common greedy strategy).
    dr, dc = tr - r, tc - c
    if abs(dr) >= abs(dc):
        if dr > 0: candidates.append(0)  # South
        if dr < 0: candidates.append(1)  # North
        if dc > 0: candidates.append(2)  # East (Right)
        if dc < 0: candidates.append(3)  # West (Left)
    else:
        if dc > 0: candidates.append(2)  # East
        if dc < 0: candidates.append(3)  # West
        if dr > 0: candidates.append(0)  # South
        if dr < 0: candidates.append(1)  # North

    # Add any missing movement actions so we have a complete fallback order (0..3).
    # This allows recovery if the best directions are blocked by walls/doors or masked out.
    for a in (0, 1, 2, 3):
        if a not in candidates:
            candidates.append(a)

    # Build the tabu set: recent positions visited in THIS phase only.
    # We do not mix pickup and dropoff histories to reduce bias across phases.
    tabu_positions = {pos for ph, pos in LAST_POSITIONS if ph == phase}

    ranked = []  # list of (cost_tuple, action)
    for a in candidates:
        # Skip actions that are not allowed by the mask (e.g., illegal moves into walls).
        if mask[a] != 1:
            continue

        nr, nc = next_pos(r, c, a)  # resulting position after taking action a
        dist = manhattan((nr, nc), dst)
        visits = VISITS[(phase, nr, nc)] #count of times this cell was visited
        is_tabu = (nr, nc) in tabu_positions

        # Construct a "cost" tuple for sorting (lexicographic ascending is better):
        #   cost[0]: 1 if position is tabu else 0 (prefer non-tabu)
        #   cost[1]: 1 if visits >= MAX_VISITS_PER_PHASE else 0 (prefer not exceeding cap)
        #   cost[2]: visit count (prefer fewer)
        #   cost[3]: Manhattan distance to the target (prefer closer)
        # With this, Python's tuple comparison will enforce our preference order naturally.
        cost = (
            1 if is_tabu else 0,
            1 if visits >= MAX_VISITS_PER_PHASE else 0,
            visits,
            dist
        )
        ranked.append((cost, a))

    if not ranked:
        # Failsafe: if everything got filtered out (should be very rare), pick any valid movement action.
        for a in (0, 1, 2, 3):
            if mask[a] == 1:
                return a
        return 0  # Extremely rare: as a last resort, pick South.

    # Choose the action with the smallest cost according to our ranking strategy.
    ranked.sort(key=lambda t: t[0])
    return ranked[0][1]


def can_pickup(r, c, pass_loc, locs, mask):
    """
    Return True if we can and should perform a Pickup action at the current cell.

    Conditions:
      - Passenger is NOT already in the taxi (pass_loc < len(locs))
      - Current (r,c) matches the passenger's location cell
      - The action mask allows Pickup (index 4)

    """
    return (pass_loc < len(locs)) and ((r, c) == locs[pass_loc]) and (mask is None or mask[4] == 1)


def can_dropoff(r, c, dest_idx, locs, in_taxi, mask):
    """
    Return True if we can and should perform a Dropoff action at the current cell.

    Conditions:
      - Passenger is inside the taxi (in_taxi == True)
      - Current (r,c) matches the destination cell
      - The action mask allows Dropoff (index 5)

    """
    return in_taxi and ((r, c) == locs[dest_idx]) and (mask is None or mask[5] == 1)


def select_action(env, obs, info):
    """
    High-level policy: choose the next action given the current observation and info.

    Logic:
      - Decode the state into (row, col, pass_loc, dest_idx)
      - Determine whether we are in "pickup" (passenger outside) or "dropoff" (passenger inside)
      - If over passenger and allowed -> Pickup
      - Else if carrying and at destination and allowed -> Dropoff
      - Else move greedily toward the current target (passenger or destination),
        while respecting the action mask and short-term memory.

    """
    r, c, pass_loc, dest_idx = decode_state(env, obs)
    locs = env.unwrapped.locs                 # Special locations list provided by Taxi env
    mask = info.get("action_mask")            # Mask with 1/0 for allowed/blocked actions
    in_taxi = (pass_loc == len(locs))         # Passenger is inside if pass_loc equals len(locs)
    phase = "dropoff" if in_taxi else "pickup"

    if not in_taxi:
        # PICKUP PHASE: Only pick up if we are exactly on the passenger's cell.
        if can_pickup(r, c, pass_loc, locs, mask):
            return 4  # Pickup
        # Otherwise, move toward the passenger's location.
        target = locs[pass_loc]
        return move_towards((r, c), target, mask, phase)
    else:
        # DROPOFF PHASE: Only drop off if we are exactly on the destination cell.
        if can_dropoff(r, c, dest_idx, locs, in_taxi, mask):
            return 5  # Dropoff
        # Otherwise, move toward the destination.
        target = locs[dest_idx]
        return move_towards((r, c), target, mask, phase)


# =============================== Main loop ================================

def main():
    """
    Entry point: create Taxi-v3 with a custom map, then run episodes indefinitely.

    Responsibilities:
      - Build env with `desc` obtained from `prepare_for_env('map_1.txt')`.
      - Reset env, initialize phase and memory structures.
      - For each step:
          * Select an action (rule-based).
          * Step the env, accumulate reward.
          * Update short-term memory (tabu/visits) and reset it when switching from pickup->dropoff.
      - On episode end (terminated/truncated), print total reward, reset memory and start next episode.

    Global variables modified:
      - LAST_POSITIONS (deque), VISITS (defaultdict), LAST_PHASE (str)
    """
    global LAST_POSITIONS, VISITS, LAST_PHASE

    # Load a custom map description - layout defined in 'map_1.txt'.
    desc = prepare_for_env('map_1.txt')

    # Create the Taxi-v3 environment.
    # render_mode="human" opens a window with simple visuals.
    env = gym.make("Taxi-v3", desc=desc, render_mode="human")

    # Reset the environment and set a deterministic seed for reproducibility of the initial state.
    obs, info = env.reset(seed=42)

    episode, total_reward = 1, 0.0
    print(f"Episode {episode} — Non ML agent + short-term memory")

    # ---- Initialize phase and memory based on the initial state ----
    r, c, pass_loc, dest_idx = decode_state(env, obs)
    locs = env.unwrapped.locs
    LAST_PHASE = phase_from(pass_loc, locs)

    # Mark the starting cell as visited in the current phase and push to the tabu queue.
    VISITS[(LAST_PHASE, r, c)] += 1
    LAST_POSITIONS.clear()
    LAST_POSITIONS.append((LAST_PHASE, (r, c)))
    # ----------------------------------------------------------------

    try:
        while True:
            # Decode current state for decision making (obs is an int).
            r, c, pass_loc, dest_idx = decode_state(env, obs)

            # Decide what to do next (movement or pickup/dropoff).
            action = select_action(env, obs, info)

            # Execute the action in the environment.
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # ---- Update short-term memory after the transition ----
            # We check if we've transitioned from pickup->dropoff (i.e., just picked up the passenger).
            r2, c2, pass_loc2, _ = decode_state(env, obs)
            new_phase = phase_from(pass_loc2, locs)

            # If the phase has switched from pickup to dropoff, we RESET the short-term memory.
            # This prevents pickup-phase biases from affecting the dropoff navigation.
            if LAST_PHASE == "pickup" and new_phase == "dropoff":
                LAST_POSITIONS.clear()
                VISITS.clear()

            # Update phase, visit counters, and tabu positions for the new state.
            LAST_PHASE = new_phase
            VISITS[(LAST_PHASE, r2, c2)] += 1
            LAST_POSITIONS.append((LAST_PHASE, (r2, c2)))
            # -------------------------------------------------------

            # If the episode ends (either success or timeout/truncation), reset for a new episode.
            if terminated or truncated:
                print(f"Episode {episode} finished — total reward: {total_reward:.0f}")

                # Full environment reset (also resets `info`); short-term memory is reset too.
                obs, info = env.reset()
                episode += 1
                total_reward = 0.0

                # Reset memory structures explicitly for the fresh episode.
                LAST_POSITIONS.clear()
                VISITS.clear()

                # Recompute initial phase and seed the memory with the starting cell.
                r, c, pass_loc, dest_idx = decode_state(env, obs)
                LAST_PHASE = phase_from(pass_loc, env.unwrapped.locs)
                VISITS[(LAST_PHASE, r, c)] += 1
                LAST_POSITIONS.append((LAST_PHASE, (r, c)))

                print(f"Episode {episode} — ready")

            # Small sleep to avoid maxing out CPU and to make human rendering smooth.
            time.sleep(0.02)

    finally:
        # Always close the environment cleanly to release rendering resources.
        env.close()


if __name__ == "__main__":
    main()