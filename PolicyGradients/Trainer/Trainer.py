import argparse
import numpy as np
import os
import time
import tensorflow as tf
import tqdm

from PolicyGradients.Trainer.Replay import get_replay_buffer


class Trainer:
    def __init__(self, agent, env, args, visualizer=None):
        assert isinstance(args, dict), "Expected args to be of type dict"
        for k, v in args.items():
            setattr(self, k, v)

        self.agent = agent
        self.env = env
        self.visualizer = visualizer
        self.replay_buffer = get_replay_buffer(agent.state_shape, agent.action_shape, self.buffer_size)

        self.output_dir = self.output_dir
        self.model_dir = self.model_dir

        # self._set_check_point(args.model_dir)

        # prepare TensorBoard output
        self.writer = tf.summary.create_file_writer(self.output_dir)
        self.writer.set_as_default()

    def env_step(self, action):
        state, reward, done, _ = self.env.step(action)
        return (
            state.astype(np.float32),
            np.array(reward, np.float32),
            np.array(done, np.int)
        )

    def _tf_env_step(self, action: tf.Tensor):
        return tf.numpy_function(self.env_step, [action], [tf.float32, tf.float32, tf.uint8])

    def __call__(self):
        total_steps, running_reward = 0, 0

        with tqdm.trange(1, self.max_episodes) as t:
            for episode in t:
                obs, steps = self.env.reset(), 0
                episode_start_time, episode_reward = time.perf_counter(), 0

                for steps in range(self.max_eps_steps):
                    if total_steps < self.n_warmup_steps:
                        action = self.env.action_space.sample()
                    else:
                        action = self.agent.get_action(obs)

                    next_obs, reward, done = self._tf_env_step(action)

                    total_steps += 1

                    if self.show_progress_intvl is not None and self.show_progress_intvl % episode == 0:
                        self.env.render()

                    self.replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done)

                    if total_steps % self.agent.policy_update_interval == 0 and total_steps > self.agent.batch_size:
                        samples = self.replay_buffer.sample(self.agent.batch_size)
                        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done = [tf.constant(samples[key])
                                                                                               for key in
                                                                                               samples.keys()]
                        self.agent.train(batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done)

                    obs = next_obs
                    episode_reward += reward

                    if done:
                        fps = steps / (time.perf_counter() - episode_start_time)
                        running_reward = .99 * running_reward + .01 * episode_reward
                        t.set_description(f"Episode {episode}")
                        t.set_postfix(episode_reward=episode_reward, running_reward=running_reward, fps=fps)

                        tf.summary.scalar(name="Common/training_reward", data=episode_reward, step=episode)
                        tf.summary.scalar(name="Common/training_episode_length", data=steps, step=episode)

                        self.replay_buffer.on_episode_end()

                        if running_reward >= 0:
                            print(f"\n Solved at episode {episode}, average reward: {running_reward:.2f}")
                            return
                        obs = self.env.reset()

        tf.summary.flush()

    def _set_check_point(self, model_dir):
        # Save and restore model
        self.checkpoint = tf.train.Checkpoint(agent=self.agent)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.output_dir, max_to_keep=5)

        if model_dir is not None:
            assert os.path.isdir(model_dir)
            self.latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            self.checkpoint.restore(self.latest_path_ckpt)

    @staticmethod
    def get_args(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')

        parser.add_argument('--max-episodes', type=int, default=int(1e4), help='Max number of episodes')
        parser.add_argument('--max-eps-steps', type=int, default=500, help='Max number of steps in an episode')
        parser.add_argument('--n-warmup-steps', type=int, default=500, help='Number of initial,exploratory actions')
        parser.add_argument('--show-progress', type=bool, default=True, help='Show agent progress during training')
        parser.add_argument('--show-progress-intvl', type=int, default=250,
                            help='Interval for showing agent progress during training')
        parser.add_argument('--buffer-size', type=int, default=int(1e6),
                            help='Number of transitions stored in replay memory')

        parser.add_argument('--output-dir', type=str, default='./Experiments', help='Directory for tensorboard output')
        parser.add_argument('--model-dir', type=str, default='./Models/', help='Directory to checkpoint models')

        return parser
