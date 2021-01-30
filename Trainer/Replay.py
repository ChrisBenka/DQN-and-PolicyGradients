from cpprb import ReplayBuffer, PrioritizedReplayBuffer


def get_replay_buffer(args):
	buffer_size, use_prioritized = args.buffer_size, args.use_prioritized
	obs_shape, act_dim = args.obs_shape, args.act_dim
	assert obs_shape is not None, "Expected obs_shape to be defined"
	assert act_dim is not None, "Expected act_dim to be defined"
	assert use_prioritized is not None, "Expected use_prioritized to be defined"

	env_dict = {
		"obs": {"shape": obs_shape},
		"act": {"shape": act_dim},
		"rew": {},
		"next_obs": {"shape": obs_shape},
		"done": {}
	}
	rb = PrioritizedReplayBuffer(buffer_size, env_dict) if use_prioritized \
		else ReplayBuffer(buffer_size, env_dict)

	return rb
