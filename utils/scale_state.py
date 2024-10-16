import numpy as np


def scale_state_to_tuple(state_dict: dict, state_scale: dict | None) -> tuple:
    if state_scale:
        return tuple(
            np.int64(state_values // state_scale.get(state_key)).item()
            for state_key, state_values in state_dict.items()
        )
    else:
        return tuple(
            np.float64(state_values).item() for state_values in state_dict.values()
        )
