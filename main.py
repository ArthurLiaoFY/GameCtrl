# %%

from config import train_kwargs
from train.collect_buffer_data import CollectBufferData
from train.train_q_table import TrainQTable

# %%
# tqt = TrainQTable(**train_kwargs)
# tqt.q_agent.load_table(model_file_path="./agent/trained_agent", prefix="flappy_bird_")
# tqt.train_agent()
# tqt.q_agent.save_table(model_file_path="./agent/trained_agent", prefix="flappy_bird_")
# %%
# tqt = TrainQTable(**train_kwargs)

# tqt.q_agent.load_table(model_file_path="./agent/trained_agent", prefix="flappy_bird_")
# tqt.inference_once(episode=0, save_animate=False)
# %%
# collect replay buffer
cbd = CollectBufferData(**train_kwargs)


# %%
