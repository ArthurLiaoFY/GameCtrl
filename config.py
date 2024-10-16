feature_scaling = {
    "next_next_pipe_bottom_y": 40,
    "next_next_pipe_dist_to_player": 512,
    "next_next_pipe_top_y": 40,
    "next_pipe_bottom_y": 10,
    "next_pipe_dist_to_player": 10,
    "next_pipe_top_y": 10,
    "player_vel": 4,
    "player_y": 16,
}

q_learning_kwargs = {
    "state_dim": 8,
    "action_dim": 2,
    "learning_rate": 0.1,
    "explore_rate": 0.5,
    "learning_rate_min": 0.03,
    "explore_rate_min": 0.03,
    "learning_rate_decay": 0.999,
    "explore_rate_decay": 0.999,
    "discount_factor": 0.99,
    "fully_explore_step": 20000,
}

ddpg_kwargs = {
    "batch_size": 256,
    "state_dim": 8,
    "action_dim": 1,
    "learning_rate": 0.1,
    "explore_rate": 0.5,
    "learning_rate_min": 0.03,
    "explore_rate_min": 0.03,
    "learning_rate_decay": 0.999,
    "explore_rate_decay": 0.999,
    "discount_factor": 0.99,
    "fully_explore_step": 0,
}

# Replay buffer
replay_buffer_kwargs = {
    "buffer_size": 1e7,
    "replay_buffer_dir": "./buffer_data",
}


train_kwargs = {
    "n_episodes": 1000,
    "inference_per_episode": 1000,
    "feature_scaling": feature_scaling,
    "ddpg_kwargs": ddpg_kwargs,
    "q_learning_kwargs": q_learning_kwargs,
    "replay_buffer_kwargs": replay_buffer_kwargs,
}
