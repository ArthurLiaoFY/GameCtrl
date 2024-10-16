import os

import numpy as np

from agent.q_table import Agent
from ple import PLE
from ple.games.flappybird import FlappyBird
from utils.make_animate import make_animate
from utils.reward_redefine import reward_redefine
from utils.scale_state import scale_state_to_tuple

os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = "dummy"


class TrainQTable:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)
        self.env = PLE(
            game=FlappyBird(),
            fps=30,
            display_screen=False,
        )
        self.q_agent = Agent(**self.q_learning_kwargs)

        self.max_train_reward = -np.inf
        self.episode_reward_traj = []

    def inference_once(self, episode: int, save_animate: bool = True):
        # set up env
        inference_reward = 0
        self.env.reset_game()
        frames = []

        # shutdown explore
        self.q_agent.shutdown_explore

        # play game
        while not self.env.game_over():
            action_idx = self.q_agent.select_action_idx(
                scale_state_to_tuple(
                    state_dict=self.env.getGameState(),
                    state_scale=self.q_learning_kwargs.get("feature_scaling"),
                )
            )
            reward = self.env.act(self.env.getActionSet()[action_idx])
            redefined_reward = reward_redefine(
                state_dict=self.env.getGameState(),
                reward=reward,
            )

            inference_reward += redefined_reward
            if save_animate:
                frames.append(self.env.getScreenRGB())

        print(f"[{episode:06d}] inference reward: {inference_reward:.4f}")

        # restart explore
        self.q_agent.start_explore

        if save_animate:
            make_animate(frames=frames, animate_name=f"animate_{episode:06d}.mp4")

    def train_once(self, episode: int):
        # set up env
        train_reward = 0
        self.env.reset_game()

        while not self.env.game_over():
            # state
            state = scale_state_to_tuple(
                state_dict=self.env.getGameState(),
                state_scale=self.q_learning_kwargs.get("feature_scaling"),
            )

            # action
            action_idx = self.q_agent.select_action_idx(state)

            # reward
            reward = self.env.act(self.env.getActionSet()[action_idx])
            next_state_dict = self.env.getGameState()
            redefined_reward = reward_redefine(
                state_dict=next_state_dict,
                reward=reward,
            )
            # cumulate reward
            train_reward += redefined_reward

            # update agent policy per step
            self.q_agent.update_policy(
                state_tuple=state,
                action_idx=action_idx,
                reward=redefined_reward,
                next_state_tuple=scale_state_to_tuple(
                    state_dict=next_state_dict,
                    state_scale=self.q_learning_kwargs.get("feature_scaling"),
                ),
            )

        # update status
        self.q_agent.update_lr_er(episode=episode)
        if train_reward > self.max_train_reward:
            print(
                f"[{episode:06d}] max_train_reward updated from {self.max_train_reward:.4f} to {train_reward:.4f}"
            )
            self.max_train_reward = train_reward

        # record reward
        self.episode_reward_traj.append(train_reward)

    def train_agent(
        self,
        save_traj_to_buffer: bool = True,
        save_network: bool = True,
    ):
        for episode in range(self.q_learning_kwargs.get("n_episodes") + 1):
            if episode % self.inference_per_episode == 0:
                self.inference_once(episode=episode)
            else:
                self.train_once(episode=episode)
