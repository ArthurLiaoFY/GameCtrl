import os

import numpy as np
import torch
from tensordict import TensorDict

from agent.dqn import Agent
from ple import PLE
from ple.games.flappybird import FlappyBird
from train.collect_buffer_data import CollectBufferData
from utils.make_animate import make_animate
from utils.reward_redefine import reward_redefine
from utils.scale_state import scale_state_to_tuple

os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = "dummy"


class TrainDQN:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)
        self.env = PLE(
            game=FlappyBird(),
            fps=30,
            display_screen=False,
        )
        self.dqn_agent = Agent(**self.dqn_kwargs)

        self.max_train_reward = -np.inf
        self.episode_reward_traj = []

    def inference_once(self, episode: int, save_animate: bool = True):
        # set up env
        inference_reward = 0
        self.env.reset_game()
        frames = []

        # shutdown explore
        self.dqn_agent.shutdown_explore

        # play game
        while not self.env.game_over():
            action_idx = self.dqn_agent.select_action_idx(
                state_tuple=scale_state_to_tuple(
                    self.env.getGameState(), state_scale=None
                )
            )
            reward = self.env.act(self.env.getActionSet()[action_idx])
            inference_reward += reward
            frames.append(self.env.getScreenRGB())

        print(f"[{episode:06d}] inference reward: {inference_reward:.4f}")

        # restart explore
        self.dqn_agent.start_explore

        if save_animate:
            make_animate(frames=frames, animate_name=f"dqn_animate_{episode:06d}.mp4")

    def train_agent(
        self,
        buffer_data: CollectBufferData,
        save_traj_to_buffer: bool = True,
        save_network: bool = True,
    ):
        for episode in range(self.n_episodes + 1):
            if episode % self.inference_per_episode == 0:
                self.inference_once(episode=episode)
            else:
                # set up env
                train_reward = 0
                self.env.reset_game()

                while not self.env.game_over():
                    # state
                    state = scale_state_to_tuple(
                        state_dict=self.env.getGameState(),
                        state_scale=None,
                    )

                    # action
                    action_idx = self.dqn_agent.select_action_idx(state)

                    # reward
                    reward = self.env.act(self.env.getActionSet()[action_idx])
                    next_state_dict = self.env.getGameState()
                    redefined_reward = reward_redefine(
                        state_dict=next_state_dict,
                        reward=reward,
                    )
                    # next state
                    next_state_tuple = scale_state_to_tuple(
                        state_dict=next_state_dict,
                        state_scale=None,
                    )

                    # cumulate reward
                    train_reward += redefined_reward

                    sample_batch = buffer_data.replay_buffer.sample(
                        batch_size=self.dqn_kwargs.get("batch_size")
                    )
                    # update agent policy per step
                    self.dqn_agent.update_policy(sample_batch=sample_batch)

                if len(state) >= 0:
                    buffer_data.replay_buffer.extend(
                        TensorDict(
                            {
                                "state": torch.Tensor(state)[None, :],
                                "action_idx": torch.Tensor([action_idx])[None, :],
                                "redefined_reward": torch.Tensor([redefined_reward])[
                                    None, :
                                ],
                                "next_state": torch.Tensor(next_state_tuple)[None, :],
                            },
                            batch_size=[1],
                        )
                    )

                # update status
                self.dqn_agent.update_lr_er(episode=episode)
                if train_reward > self.max_train_reward:
                    print(
                        f"[{episode:06d}] max_train_reward updated from {self.max_train_reward:.4f} to {train_reward:.4f}"
                    )
                    self.max_train_reward = train_reward

                # record reward
                self.episode_reward_traj.append(train_reward)
