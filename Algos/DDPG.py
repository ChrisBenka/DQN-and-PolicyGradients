import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers


class Actor(tf.keras.Model):
	def __init__(self, action_shape, action_upper_bound,
	             action_lower_bound, hidden_units=(400, 300), name='Actor'):
		super(Actor, self).__init__(name=name)
		self.action_upper_bound = action_upper_bound
		self.action_lower_bound = action_lower_bound

		self.fc1 = Dense(hidden_units[0], activation='relu', name='fc1')
		self.fc2 = Dense(hidden_units[1], activation='relu', name='fc2')
		self.fc3 = Dense(action_shape, activation='tanh', name='fc3')

	def call(self, inputs, training=None, mask=None):
		x = self.fc1(inputs)
		x = self.fc2(x)
		x = self.fc3(x)
		action = tf.multiply(x, self.action_upper_bound)
		return action

	def get_config(self):
		pass


class Critic(tf.keras.Model):
	def __init__(self, hidden_units=(400, 300), name='Critic'):
		super(Critic, self).__init__(name=name)

		self.fc1 = Dense(hidden_units[0], activation='relu', name='fc1')
		self.fc2 = Dense(hidden_units[1], activation='relu', name='fc2')
		self.fc3 = Dense(1, name='fc3')

	def call(self, inputs, training=None, mask=None):
		x = self.fc1(inputs)
		x = self.fc2(x)
		x = self.fc3(x)
		action = tf.multiply(x, self.action_upper_bound)
		return action

	def get_config(self):
		pass


def _update_target(target_weights, weights, tau):
	for (a, b) in zip(target_weights, weights):
		a.assign(b * tau + a * (1 - tau))


class DDPG:
	def __init__(self, state_shape, action_shape,
	             action_lower_bound, action_upper_bound,
	             actor_hidden_units=(400, 300),
	             critic_hidden_units=(400, 300),
	             actor_lr=.05, critic_lr=.05,
	             noise_mu=0, noise_std=.2, tau=.5,
	             noise_theta=.15, noise_dt=1e-2,
	             l2_reg_actor=0, lr_reg_critic=0,
	             normalize_rewards=False, **kwargs):
		self._state_shape = state_shape
		self._action_shape = action_shape
		self._action_lower_bound = action_lower_bound
		self._action_upper_bound = action_upper_bound

		self._actor = Actor(action_shape, action_upper_bound, action_lower_bound, actor_hidden_units)
		self._target_actor = Actor(action_shape, action_upper_bound, action_lower_bound, actor_hidden_units)
		self._actor_optimizer = optimizers.Adam(learning_rate=actor_lr)

		self._critic = Critic(critic_hidden_units)
		self._target_critic = Critic(state_shape, )
		self._critic_optimizer = optimizers.Adam(learning_rate=critic_lr)

		self._noise_mu = noise_mu
		self._noise_std = noise_std

		self._tau = tau

	def _cal_td_error(self, states, actions, rewards, next_states, dones, gamma=.99):
		target_actions = self._target_actor(next_states)
		not_dones = 1 - tf.cast(dones, tf.float32)
		td_target = tf.cast(rewards, tf.float32) + tf.cast(gamma, tf.float32) * \
		            self._target_critic([next_states, target_actions]) * not_dones
		td_pred = self._critic([states, actions], training=True)
		td_error = tf.stop_gradient(td_target) - td_pred
		return td_error

	def train(self, batch_obs: tf.Tensor, batch_actions: tf.Tensor, batch_rewards: tf.Tensor, batch_next_obs: tf.Tensor,
	          batch_done: tf.Tensor):
		with tf.GradientTape() as tape:
			td_error = self._cal_td_error(batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done, self._gamma)
			critic_loss = tf.reduce_mean(td_error ** 2)
		critic_grads = tape.gradient(critic_loss, self._critic.trainable_variables)

		self._critic_optimizer.apply_gradients(
			zip(critic_grads, self._critic.trainable_variables)
		)

		with tf.GradientTape() as tape:
			sampled_actions = self._actor(batch_obs)
			actor_loss = -tf.reduce_mean(
				self._critic([batch_obs, sampled_actions])
			)
		actor_grads = tape.gradient(
			actor_loss, self._actor.trainable_variables
		)

		self._actor_optimizer.apply_gradients(
			zip(actor_grads, self._actor.trainable_variables)
		)

		_update_target(self._target_actor,self._actor,self._tau)
		_update_target(self._target_actor,self._critic,self._tau)

	def get_action(self, obs: tf.Tensor):
		action = tf.squeeze(self._actor(obs[None]))
		random_noise = np.random.normal(self._noise_std, self._noise_std, self._state_shape)
		action = action + tf.constant(random_noise, dtype=tf.float32)
		valid_action = tf.clip_by_value(action, self._action_lower_bound,
		                                self._action_upper_bound)
		return valid_action


	@staticmethod
	def get_args(parser=None):
		pass