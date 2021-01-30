import os
import tensorflow as tf
import time


class Trainer:
	def __init__(self, agent, env, visualizer, args, output_dir='./Experiments/', model_dir='./Models/'):
		assert isinstance(args, dict), "Expected args to be of type dict"
		for k, v in args:
			setattr(self, k, v)

		self._agent = agent
		self._env = env
		self._visualizer = visualizer
		self._replay_buffer = get_replay_buffer(env, args)

		self._output_dir = output_dir
		self._model_dir = model_dir

		if args.evaluate:
			assert args.model_dir is not None
		self._set_check_point(args.model_dir)

		# prepare TensorBoard output
		self.writer = tf.summary.create_file_writer(self._output_dir)
		self.writer.set_as_default()

	def __call_(self):
		assert self._num_episodes is not None, "Expected _num_episodes to be defined"
		assert self._max_eps_steps is not None, "Expected _max_eps_steps to be defined"

		total_steps = 0

		for episode in self._num_episodes:
			obs, steps = self._env.reset(), 0
			episode_start_time, episode_return = time.perf_counter(), 0

			for steps in range(self._max_eps_steps):
				if total_steps < self._n_warmup_steps:
					action = self._env.action_space.sample()
				else:
					action = self._agent.get_action(obs)

				next_obs, reward, done, _ = self._env.step(action)

				if self._show_progress_intvl is not None and self._show_progress_intvl % episode == 0:
					self._env.render()

				if self._replay_buffer is not None:
					self._replay_buffer.store_transitions(obs, action, next_obs, reward, done)

					if total_steps % self._agent.policy_update_interval == 0:
						samples = self._replay_buffer.sample(self._agent.batch_size)
						self._agent.train(samples)

				elif total_steps % self._agent.policy_update_interval == 0:
					self._agent.train()

				obs = next_obs

				if done:
					fps = steps / (time.perf_counter() - episode)
					self.logger.info(
						"Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
							episode, total_steps, steps, episode_return, fps))
					tf.summary.scalar(name="Common/training_return", data=episode_return)
					tf.summary.scalar(name="Common/training_episode_length", data=steps)

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











