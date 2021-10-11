import argparse

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense


class Actor(tf.keras.Model):
    def __init__(self, action_shape, action_upper_bound,
                 action_lower_bound, hidden_units=(400, 300), name='Actor'):
        super(Actor, self).__init__(name=name)
        self.action_upper_bound = action_upper_bound
        self.action_lower_bound = action_lower_bound

        self.fc1 = Dense(hidden_units[0], activation='relu', name='fc1')
        self.fc2 = Dense(hidden_units[1], activation='relu', name='fc2')
        self.fc3 = Dense(action_shape[0], activation='tanh', name='fc3')

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
        [states, actions] = inputs
        x = self.fc1(tf.concat([states, actions], axis=1))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

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
                 noise_mu=0, noise_std=.2, tau=.005,
                 noise_theta=.15, noise_dt=1e-2,
                 batch_size=32, policy_update_interval=5,
                 l2_reg_actor=0, lr_reg_critic=0,
                 normalize_rewards=False, **kwargs):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound = action_upper_bound

        self.batch_size = batch_size
        self.policy_update_interval = policy_update_interval

        self.actor = Actor(action_shape, action_upper_bound, action_lower_bound, actor_hidden_units)
        self.target_actor = Actor(action_shape, action_upper_bound, action_lower_bound, actor_hidden_units)
        self.actor_optimizer = optimizers.Adam(learning_rate=actor_lr)

        self.critic = Critic(critic_hidden_units)
        self.target_critic = Critic(critic_hidden_units)
        self.critic_optimizer = optimizers.Adam(learning_rate=critic_lr)

        self.noise_mu = noise_mu
        self.noise_std = noise_std

        self.tau = tau

    def _cal_td_error(self, states, actions, rewards, next_states, dones, gamma=.99):
        target_actions = self.target_actor(next_states)
        not_dones = 1 - tf.cast(dones, tf.float32)
        td_target = tf.cast(rewards, tf.float32) + tf.cast(gamma, tf.float32) * \
                    self.target_critic([next_states, target_actions]) * tf.cast(not_dones, tf.float32)
        td_pred = self.critic([states, actions], training=True)
        td_error = tf.stop_gradient(td_target) - td_pred
        return td_error

    @tf.function
    def train(self, batch_obs: tf.Tensor, batch_actions: tf.Tensor, batch_rewards: tf.Tensor, batch_next_obs: tf.Tensor,
              batch_done: tf.Tensor):
        with tf.GradientTape() as tape:
            td_error = self._cal_td_error(batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done)
            critic_loss = tf.reduce_mean(td_error ** 2)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            sampled_actions = self.actor(batch_obs)
            actor_loss = -tf.reduce_mean(
                self.critic([batch_obs, sampled_actions])
            )
        actor_grads = tape.gradient(
            actor_loss, self.actor.trainable_variables
        )

        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables)
        )

        _update_target(self.target_actor.trainable_variables, self.actor.trainable_weights, self.tau)
        _update_target(self.target_critic
                       .trainable_variables, self.critic.trainable_variables, self.tau)

    def get_action(self, obs: tf.Tensor):
        action = tf.squeeze(self.actor(obs[None]))
        action = action + tf.random.normal(mean=self.noise_mu, stddev=self.noise_std, shape=self.action_shape,
                                           dtype=tf.float32)
        valid_action = tf.clip_by_value(action, self.action_lower_bound,
                                        self.action_upper_bound)
        return valid_action

    @staticmethod
    def get_args(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        parser.add_argument('--actor-hidden-units', type=tuple, default=[400, 300], help='actor-hidden-units')
        parser.add_argument('--critic-hidden-units', type=tuple, default=[400, 300], help='critic-hidden-units')
        parser.add_argument('--actor-lr', type=float, default=.005, help='actor learning rate')
        parser.add_argument('--critic-lr', type=float, default=.005, help='critic learning rate')
        parser.add_argument('--noise-mu', type=float, default=0, help='noise mu')

        return parser
