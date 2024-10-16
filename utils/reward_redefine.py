def reward_redefine(state_dict: dict, reward: float):
    if reward == 0:
        if state_dict.get("player_y") > state_dict.get(
            "next_pipe_top_y"
        ) or state_dict.get("player_y") < state_dict.get("next_pipe_bottom_y"):
            return (
                -0.05
                / 512  # screen height
                * min(
                    abs(state_dict.get("player_y") - state_dict.get("next_pipe_top_y")),
                    abs(
                        state_dict.get("player_y")
                        - state_dict.get("next_pipe_bottom_y")
                    ),
                )
            )
        else:
            return 0.1 / 512

    else:
        return float(reward)
