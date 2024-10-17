import os

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer
from tqdm import tqdm

from agent.q_table import Agent
from ple import PLE
from ple.games.flappybird import FlappyBird
from utils.reward_redefine import reward_redefine
from utils.scale_state import scale_state_to_tuple

os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = "dummy"


class CollectBufferData:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)
        self.env = PLE(
            game=FlappyBird(),
            fps=30,
            display_screen=False,
        )
        self.expert_agent = Agent(**self.q_learning_kwargs)
        self.expert_agent.load_table(
            model_file_path="./agent/trained_agent", prefix="flappy_bird_"
        )
        self.expert_agent.shutdown_explore

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.replay_buffer_kwargs.get("buffer_size"),
            )
        )
        self.load_replay_buffer()

    def extend_buffer_data(self, extend_amount: int, save: bool = True) -> None:
        for _ in tqdm(range(extend_amount)):
            self.env.reset_game()
            state_list = []
            action_idx_list = []
            reward_list = []
            next_state_list = []
            while not self.env.game_over():
                state_tuple = scale_state_to_tuple(
                    state_dict=self.env.getGameState(),
                    state_scale=self.feature_scaling,
                )
                state_list.append(state_tuple)

                action_idx = self.expert_agent.select_action_idx(
                    state_tuple=state_tuple
                )
                reward = self.env.act(action=self.env.getActionSet()[action_idx])
                next_state_dict = self.env.getGameState()
                redefined_reward = reward_redefine(
                    state_dict=next_state_dict, reward=reward
                )

                action_idx_list.append(action_idx)
                reward_list.append(redefined_reward)
                next_state_list.append(
                    scale_state_to_tuple(
                        state_dict=next_state_dict,
                        state_scale=self.feature_scaling,
                    )
                )
            if len(state_list) > 0:
                self.replay_buffer.extend(
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

        if save:
            self.save_replay_buffer()

    def save_replay_buffer(self) -> None:
        print(
            "buffer data save to dir: {replay_buffer_dir}".format(
                replay_buffer_dir=self.replay_buffer_kwargs.get("replay_buffer_dir")
            )
        )
        self.replay_buffer.dumps(self.replay_buffer_kwargs.get("replay_buffer_dir"))

    def load_replay_buffer(self) -> None:

        try:
            self.replay_buffer.loads(self.replay_buffer_kwargs.get("replay_buffer_dir"))
            print(
                "buffer data load from dir: {replay_buffer_dir}".format(
                    replay_buffer_dir=self.replay_buffer_kwargs.get("replay_buffer_dir")
                )
            )

        except FileNotFoundError:
            pass
