import os

import numpy as np
import torch
from tensordict import TensorDict

from agent.double_dqn import Agent
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
        self.dqn_agent = Agent(**self.ddqn_kwargs)

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
        for episode in range(self.ddqn_kwargs.get("n_episodes") + 1):
            if episode % self.inference_per_episode == 0:
                self.inference_once(episode=episode)
            else:
                # set up env
                train_reward = 0
                self.env.reset_game()

                state_list = []
                action_idx_list = []
                reward_list = []
                next_state_list = []

                while not self.env.game_over():
                    # state
                    state_tuple = scale_state_to_tuple(
                        state_dict=self.env.getGameState(),
                        state_scale=None,
                    )
                    state_list.append(state_tuple)

                    # action
                    action_idx = self.dqn_agent.select_action_idx(state_tuple)
                    action_idx_list.append(action_idx)

                    # reward
                    reward = self.env.act(self.env.getActionSet()[action_idx])
                    next_state_dict = self.env.getGameState()
                    redefined_reward = reward_redefine(
                        state_dict=next_state_dict,
                        reward=reward,
                    )
                    reward_list.append(redefined_reward)

                    # next state
                    next_state_tuple = scale_state_to_tuple(
                        state_dict=next_state_dict,
                        state_scale=None,
                    )
                    next_state_list.append(next_state_tuple)

                    # cumulate reward
                    train_reward += redefined_reward

                    sample_batch = buffer_data.replay_buffer.sample(
                        batch_size=self.ddqn_kwargs.get("batch_size")
                    )
                    # update agent policy per step
                    self.dqn_agent.update_policy(
                        episode=episode, sample_batch=sample_batch
                    )

                if len(state_list) > 0:
                    buffer_data.replay_buffer.extend(
                        TensorDict(
                            {
                                "state": torch.Tensor(np.array(state_list)),
                                "action_idx": torch.Tensor(np.array(action_idx_list))[
                                    :, None
                                ],
                                "reward": torch.Tensor(np.array(reward_list))[:, None],
                                "next_state": torch.Tensor(np.array(next_state_list)),
                            },
                            batch_size=[len(state_list)],
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
