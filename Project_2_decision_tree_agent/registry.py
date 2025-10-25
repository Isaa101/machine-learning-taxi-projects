#######################
#     registry.py     #
#######################
# Author: Enrique Mateos Melero
# File to register new implementation of environment


import gymnasium as gym
from gymnasium.envs.toy_text.taxi import TaxiEnv
from gymnasium.envs.registration import register

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "| | : | : |",
    "+---------+",
]

register(
    id="Taxi-v3",
    entry_point="gymnasium.envs.toy_text.taxi:TaxiEnv",
    kwargs={"desc": MAP},
    max_episode_steps=200
)
