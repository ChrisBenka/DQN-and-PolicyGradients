from cpprb import ReplayBuffer, PrioritizedReplayBuffer


def get_replay_buffer(obs_shape, action_dim, buffer_size=int(1e6), use_prioritized=False):
    env_dict = {
        "obs": {"shape": obs_shape},
        "act": {"shape": action_dim},
        "rew": {},
        "next_obs": {"shape": obs_shape},
        "done": {}
    }
    rb = PrioritizedReplayBuffer(buffer_size, env_dict) if use_prioritized \
        else ReplayBuffer(buffer_size, env_dict)

    return rb
