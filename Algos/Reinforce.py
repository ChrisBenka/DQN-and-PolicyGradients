import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from PolicyGradients.Utils.EnvUtils import tf_env_step
from typing import Tuple
import numpy as np

eps = np.finfo(np.float32).eps.item()


def run_episode(
    env,
    initial_state: tf.Tensor,
    policy: tf.keras.Model,
    max_steps: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    action_probs = tf.TensorArray(tf.float32, 0, True)
    rewards = tf.TensorArray(tf.int32, 0, True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        state = tf.expand_dims(state, 0)

        action_logits_t = policy(state)

        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        action_probs = action_probs.write(t, action_probs_t[0, action])

        state, reward, done = tf_env_step(env, action)

        state.set_shape(initial_state_shape)

        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    rewards = rewards.stack()

    return action_probs, rewards


def get_expected_return(rewards: tf.Tensor, gamma: float,
                        standardize: bool = True) -> tf.Tensor:
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + eps))
    return returns


def compute_loss(action_probs: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
    action_log_probs = tf.math.log(action_probs)
    loss = -tf.math.reduce_sum(action_log_probs * returns)
    return loss


class Policy(tf.keras.Model):
    def __init__(self, n_actions: int):
        super(Policy, self).__init__(name='Policy')
        self.fc1 = Dense(6, activation='relu')
        self.fc2 = Dense(n_actions)

    def call(self, inputs: tf.Tensor):
        x = self.fc1(inputs)
        action_logits = self.fc2(x)
        return action_logits


class Reinforce:
    def __init__(self, n_actions, lr=.001):
        self.policy = Policy(n_actions)
        self.optim = Adam(learning_rate=lr)

    @tf.function
    def train_step(self, env, initial_state: tf.Tensor, gamma: float,
                   max_steps_per_episode: int) -> tf.Tensor:
        with tf.GradientTape() as tape:
            action_probs, rewards = run_episode(env, initial_state, gamma,
                                                max_steps_per_episode)

            returns = get_expected_return(rewards, gamma)

            action_probs, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, returns]
            ]

          loss = compute_loss(action_probs, returns)
        grads = tape.gradient(loss,self.policy.trainable_variables)
        self.optim.apply_gradients(zip(loss,self.policy.trainable_variables))
        episode_reward = tf.math.reduce_sum(rewards)
        return episode_reward
