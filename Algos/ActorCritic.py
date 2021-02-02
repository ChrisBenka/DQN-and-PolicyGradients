from typing import List

import numpy as np
import tensorflow as tf
from PolicyGradients.Utils.EnvUtils import tf_env_step
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

eps = np.finfo(np.float32).eps.item()


def run_episodestep(
  env,
  initial_state: tf.Tensor,
  policy: tf.keras.Model,
  critic: tf.Keras.Model,
  max_steps: int
) -> List[tf.Tensor]:
    initial_state_shape = initial_state.shape
    state = initial_state

    state = tf.expand_dims(state, 0)

    action_logits_t = policy(state)

    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs = tf.nn.softmax(action_logits_t)

    value = critic(state)

    state, reward, done = tf_env_step(env, action)

    state.set_shape(initial_state_shape)

    next_value = critic(next_state)

    return next_state, done, action_probs, value, next_value, rewards


def compute_loss(action_probs: tf.Tensor, value: tf.Tensor, gamma: float
    next_value: tf.Tensor

, reward: tf.Tensor,
done: tf.Tensor) -> tf.Tensor:
advs = tf.stop_gradients(values - reward + tf.cast(gamma, tf.float32) *
                         next_value * (1 - done))
loss = -tf.math.reduce_sum(action_log_probs * advs)
return loss

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


class Policy(tf.keras.Model):
    def __init__(self, n_actions: int):
        super(Policy, self).__init__(name='Policy')
        self.fc1 = Dense(6, activation='relu')
        self.fc2 = Dense(n_actions)

    def call(self, inputs: tf.Tensor):
        x = self.fc1(inputs)
        action_logits = self.fc2(x)
        return action_logits


class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__(name='Critic')
        self.fc1 = Dense(6, activation='relu')
        self.fc2 = Dense(1)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.fc1(inputs)
        return self.fc2(x)


class ActorCritic:
    def __init__(self, n_actions, p_lr=.001, c_lr=.001):
        self.policy = Policy(n_actions=n_actions)
        self.critic = Critic()
        self.p_optim = Adam(learning_rate=p_lr)
        self.c_optim = Adam(learning_rate=c_lr)

    @tf.function
    def train_step(self, env, initial_state: tf.Tensor, gamma: float,
                   max_steps_per_episode: int) -> tf.Tensor:

        episode_reward = tf.constant(0)
        for t in range(max_steps_per_episode):
            with tf.GradientTape() as tape, tf.GradientTape() as tape2:
                results = run_episode_step(env, initial_state,
                                           gamma, max_steps_per_episode)

                next_state, done, action_probs, value, next_value, reward = results

                done, action_probs, value, next_value, reward = [
                    tf.expand_dims(x, 1) for x in [done, action_probs, value,
                                                   next_value, reward]
                ]

            actor_loss = compute_loss(action_probs, value, next_value, reward, done)
            critic_loss = huber_loss(values, reward + tf.cast(gamma, tf.float32) * next_value)

        actor_grads = tape.gradient(actor_loss, self.policy.trainable_variables)
        critic_loss = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.p_optim.apply_gradients(zip(loss, self.policy.trainable_variables))
        self.c_optim.apply_gradients(zip(loss, self.policy.trainable_variables))
        episode_reward += tf.reduce_sum(reward)

        if tf.cast(done, tf.bool):
            return episode_reward
