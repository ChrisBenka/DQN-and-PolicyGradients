import argparse
import numpy as np
import os
import tensorflow as tf
import time
import tqdm
from Replay import get_replay_buffer


class Trainer:
	def __init__(self, agent, env, visualizer, args):
		assert isinstance(args, dict), "Expected args to be of type dict"
		for k, v in args:
			setattr(self, k, v)

		self._agent = agent
		self._env = env
		self._visualizer = visualizer
		self._replay_buffer = get_replay_buffer(args)

		self._output_dir = self._output_dir
		self._model_dir = self._model_dir

		self._set_check_point(args.model_dir)

		# prepare TensorBoard output
		self.writer = tf.summary.create_file_writer(self._output_dir)
		self.writer.set_as_default()

	def env_step(self,action):
		state, reward, done = self._env.step(action)
		return (
			state.astype(np.float32),
			np.array(reward,np.float32),
			np.array(done,np.int)
		)

	def _tf_env_step(self,action: tf.Tensor):
		return tf.numpy_function(self.env_step, [action], [tf.float32, tf.float32, tf.uint8])

	def __call_(self):
		assert self._num_episodes is not None, "Expected _num_episodes to be defined"
		assert self._max_eps_steps is not None, "Expected _max_eps_steps to be defined"

		total_steps, running_reward = 0, 0

		with tqdm.trange(self._num_episodes) as t:
			for episode in t:
				obs, steps = self._env.reset(), 0
				episode_start_time, episode_reward = time.perf_counter(), 0

				for steps in range(self._max_eps_steps):
					if total_steps < self._n_warmup_steps:
						action = self._env.action_space.sample()
					else:
						action = self._agent.get_action(obs)

					next_obs, reward, done, _ = self._tf_env_step(action)

					if self._show_progress_intvl is not None and self._show_progress_intvl % episode == 0:
						self._env.render()

					self._replay_buffer.add(obs, action, next_obs, reward, done)

					if total_steps % self._agent.policy_update_interval == 0:
						samples = self._replay_buffer.sample(self._agent.batch_size)
						batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done = samples
						self._agent.train(batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done)

					obs = next_obs

					if done:
						fps = steps / (time.perf_counter() - episode)
						running_reward = .99 * running_reward + .01 * episode_reward
						t.set_description(f"Episode {episode}")
						t.set_postfix(episode_reward=episode_reward, running_reward=running_reward, fps=fps)

						tf.summary.scalar(name="Common/training_reward", data=episode_reward)
						tf.summary.scalar(name="Common/training_episode_length", data=steps)

						self._replay_buffer.on_episode_end()

						if running_reward >= self._reward_threshold:
							print(f"\n Solved at episode {i}, average reward: {running_reward:.2f}")
							return

						obs = self._env.reset()
					if self._max_steps is not None and self._max_steps >= total_steps:
						tf.summary.flush()
						return

					if self._visualize:
						if self._visualizer is not None:
							self._visualizer.visualize()
						else:
							# todo implement default visualizer
							pass

		tf.summary.flush()

	def _set_check_point(self, model_dir):
		# Save and restore model
		self._checkpoint = tf.train.Checkpoint(agent=self._agent)
		self.checkpoint_manager = tf.train.CheckpointManager(
			self._checkpoint, directory=self._output_dir, max_to_keep=5)

		if model_dir is not None:
			assert os.path.isdir(model_dir)
			self._latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
			self._checkpoint.restore(self._latest_path_ckpt)

	@staticmethod
	def get_args(parser=None):
		if parser is None:
			parser = argparse.ArgumentParser(conflict_handler='resolve')

		parser.add_argument('--max-episodes', type=int, default=int(1e4), help='Max number of episodes')
		parser.add_argument('--max-eps-steps', type=int, default=500, help='Max number of steps in an episode')
		parser.add_argument('--n_warmup_steps', type=int, default=500, help='Number of initial,exploratory actions')
		parser.add_argument('--show_progress', type=bool, default=True, help='Show agent progress during training')
		parser.add_argument('--show_progress_intvl', type=int, default=250, help='Interval for showing agent progress during training')

		parser.add_argument('--output_dir', type=str, default='./Experiments', help='Directory for tensorboard output')
		parser.add_argument('--model_dir', type=str, default='./Models/', help='Directory to checkpoint models')

		return parser
