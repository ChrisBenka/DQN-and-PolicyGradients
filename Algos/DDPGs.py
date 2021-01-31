# from abc import ABC
#
# import tensorflow as tf
# import gym
# import tqdm
# import numpy as np
# from tensorflow.keras.layers import Dense
# from tensorflow.keras import optimizers
# from PolicyGradients.StochasticProcesses.OUNoise import OUActionNoise
# from typing import List
#
#
# class Actor(tf.keras.Model):
#     def __init__(self, state_shape, action_shape, action_upper_bound,
#                  hidden_units=(400, 300), name='Actor'):
#         super(Actor, self).__init__(name=name)
#         self.state_shape = state_shape
#         self.action_shape = action_shape
#         self.action_upper_bound = action_upper_bound
#
#         self.fc1 = Dense(hidden_units[0], activation='relu', name='fc1')
#         self.fc2 = Dense(hidden_units[1], activation='relu', name='fc2')
#         self.fc3 = Dense(action_shape[0], activation='tanh', name='fc3')
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.fc1(inputs)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         action = tf.multiply(x, self.action_upper_bound)
#         return action
#
#
# class Critic(tf.keras.Model):
#     def __init__(self, state_shape, action_shape,
#                  hidden_units=(400, 300), name='Critic'):
#         super(Critic, self).__init__(name=name)
#         self.state_shape = state_shape
#         self.action_shape = action_shape
#
#         self.fc1 = Dense(hidden_units[0], activation='relu', name='fc1')
#         self.fc2 = Dense(hidden_units[1], activation='relu', name='fc2')
#         self.fc3 = Dense(1, name='fc3')
#
#     def call(self, inputs, training=None, mask=None):
#         states, actions = inputs
#         x = tf.concat((states, actions), axis=1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         v = self.fc3(x)
#         return v
#
#
# def update_target(target_weights, weights, tau):
#     for (a, b) in zip(target_weights, weights):
#         a.assign(b * tau + a * (1 - tau))
#
#
# class DDPG:
#     def __init__(self,state_shape, action_shape,
#                  action_lower_bound, action_upper_bound,
#                  actor_hidden_units=(400, 300),
#                  critic_hidden_units=(400, 300),
#                  actor_lr=.05, critic_lr=.05,
#                  noise_mu=0, noise_std=.2, tau=.5,noise_theta=.15,
#                  noise_dt=1e-2,l2_reg_actor=0, l2_reg_critic=0,
#                  normalize_rewards=True):
#
#         self.env = env
#         self.state_shape = state_shape
#         self.action_shape = action_shape
#         self.action_lower_bound = action_lower_bound
#         self.action_upper_bound = action_upper_bound
#
#         self.tau = tau
#
#         self.actor = Actor(state_shape, action_shape, action_upper_bound,
#                            actor_hidden_units)
#         self.actor_target = Actor(state_shape, action_shape,
#                                   action_upper_bound,
#                                   actor_hidden_units, name='Actor_target')
#         self.actior_optim = optimizers.Adam(learning_rate=actor_lr)
#
#         self.critic = Critic(state_shape, action_shape, critic_hidden_units)
#         self.critic_target = Critic(state_shape, action_shape,
#                                     critic_hidden_units, name='critic_target')
#         self.critic_optim = optimizers.Adam(learning_rate=critic_lr)
#
#         self.noise = OUActionNoise(noise_mu * np.ones(1) , noise_std * np.ones(1))
#
#
#
#     def get_action(self, state, training=False):
#         action = tf.squeeze(self.actor(state[None]))
#         random_noise = self.noise()
#         action = action + tf.constant(random_noise,dtype=tf.float32)
#
#         valid_action = tf.clip_by_value(action, self.action_lower_bound,
#                                         self.action_upper_bound)
#         return valid_action
#
#     def _calc_td_error(self, states, actions, next_states,
#                         rewards, dones, gamma):
#         target_actions = self.actor_target(next_states)
#         not_dones = 1 - tf.cast(dones, tf.float32)
#         td_target = tf.cast(rewards,tf.float32) + tf.cast(gamma,tf.float32) *  \
#             self.critic_target([next_states, target_actions]) * not_dones
#         td_pred = self.critic([states, actions], training=True)
#         td_error = tf.stop_gradient(td_target) - td_pred
#         return td_error
#
#     @tf.function
#     def learn(self,states,actions,rewards,next_states,dones):
#         with tf.GradientTape() as tape:
#             td_error = self.calc_td_error(
#                 states, actions, next_states, rewards, dones, gamma)
#             critic_loss = tf.reduce_mean(td_error ** 2)
#         critic_grads = tape.gradient(
#             critic_loss, self.critic.trainable_variables
#         )
#         self.critic_optim.apply_gradients(
#             zip(critic_grads, self.critic.trainable_variables)
#         )
#
#         with tf.GradientTape() as tape:
#             sampled_actions = self.actor(states)
#             actor_loss = -tf.reduce_mean(
#                 self.critic([states, sampled_actions])
#             )
#         actor_grads = tape.gradient(
#             actor_loss, self.actor.trainable_variables
#         )
#         self.actior_optim.apply_gradients(
#             zip(actor_grads, self.actor.trainable_variables)
#         )
#
#     def train_step(self, initial_state, max_steps, gamma):
#         episode_reward = tf.constant(0,dtype=tf.float32)
#         state = initial_state
#         state_shape = initial_state.shape
#
#         for t in tf.range(max_steps):
#             action = self.policy(state)
#             next_state, reward, done = self.tf_env_step(action)
#
#             episode_reward += reward
#
#             reward, done = [
#                 tf.expand_dims(x, 0) for x in [reward, done]
#             ]
#
#             state.set_shape(state_shape)
#             next_state.set_shape(state_shape)
#
#             self.replay_mem.store(state, action,reward,next_state, done)
#
#             transitions_stored = tf.constant(self.replay_mem.num_in_memory)
#
#             if tf.greater_equal(transitions_stored,
#                                 tf.constant(self.replay_batch_size)):
#
#                 batch = self.replay_mem.sample(self.replay_batch_size)
#
#                 (states, actions, rewards, next_states, dones) = batch
#
#                 self.learn(states,actions,rewards,next_states,dones)
#
#                 update_target(self.actor_target.trainable_variables, self.actor.trainable_variables, self.tau)
#                 update_target(self.critic_target.trainable_variables, self.critic.trainable_variables, self.tau)
#
#             if tf.cast(done, tf.bool):
#                 break
#
#             state = next_state
#
#         return episode_reward
#
# # if __name__ == '__main__':
# #     env = gym.make('Pendulum-v0')
# #     state_shape = env.observation_space.shape
# #     action_shape = env.action_space.shape
# #
# #     action_low_bound = env.action_space.low
# #     action_high_bound = env.action_space.high
# #
# #     ddpg = DDPG(
# #         env,state_shape,action_shape,action_low_bound,action_high_bound
# #     )
# #
# #     running_reward = 0
# #     reward_threshold = 195
# #     transitions_stored = tf.constant(0)
# #     max_episodes = 100
# #     max_steps = 200
# #     gamma = .99
# #
# #     with tqdm.trange(max_episodes) as t:
# #         rewards = []
# #         for i in t:
# #             state = tf.constant(env.reset(), dtype=tf.float32)
# #             episode_reward = float(ddpg.train_step(state, max_steps, gamma))
# #
# #             rewards.append(episode_reward)
# #
# #             running_reward = np.mean(rewards[-100:])
# #
# #             t.set_description(f"Episode {i}")
# #             t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)
# #
# #             if running_reward > reward_threshold:
# #                 break
# #
