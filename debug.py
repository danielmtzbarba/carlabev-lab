from CarlaBEV.envs import CarlaBEV
import pygame
import gymnasium as gym
import torch
from src.envs.carlabev import make_carlabev_env
from src.agents import QNetwork

device = "cuda:0"
size = 128
model_path = "runs/dqn-gridworld-seed_1-bs_20000/dqn-gridworld.cleanrl_model"
LOAD_MODEL = False

dummyenv = gym.vector.SyncVectorEnv(
    [
        make_carlabev_env(
            env_id="carlabev",
            seed=0,
            idx=1,
            capture_video=False,
            run_name="CarlaBEV",
            size=size,
        )
        for i in range(1)
    ]
)

# Initialise the environment
# env = gym.make("CarRacing-v2", render_mode="human", continuous=False)
env = CarlaBEV(size=size, render_mode="human")

model = QNetwork(dummyenv)
del dummyenv

if LOAD_MODEL:
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )

model.eval()


# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
total_reward = 0
running = True
while running:
    action = 0
    ################################# CHECK PLAYER INPUT #################################
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action = 1
            elif event.key == pygame.K_RIGHT:
                action = 2
            elif event.key == pygame.K_UP:
                action = 3
            elif event.key == pygame.K_DOWN:
                action = 4

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()
        total_reward = 0

env.close()
