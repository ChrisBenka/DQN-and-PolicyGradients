import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from PolicyGradients.Utils.EnvUtils import tf_env_step
from PolicyGradients.Algos.Reinforce import get_expected_return
from typing import List
import numpy as np

eps = np.finfo(np.float32).eps.item()


def run_episode(
    env,
    initial_state: tf.Tensor,
    policy: tf.keras.Model,
    critic: tf.Keras.Model,
    max_steps: int
) -> List[tf.Tensor]:
    action_probs = tf.TensorArray(tf.float32, 0, True)
    values = tf.TensorArray(tf.float32, 0, True)
    rewards = tf.TensorArray(tf.int32, 0, True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        state = tf.expand_dims(state, 0)

        action_logits_t = policy(state)

        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        action_probs = action_probs.write(t, action_probs_t[0, action])

        value = critic(state)

        values.write(t, value)

        state, reward, done = tf_env_step(env, action)

        state.set_shape(initial_state_shape)

        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards


def compute_loss(action_probs: tf.Tensor, values: tf.Tensor,
                 returns: tf.Tensor) -> tf.Tensor:
    advs = tf.stop_gradients(returns-values)
    action_log_probs = tf.math.log(action_probs)

    loss = -tf.math.reduce_sum(action_log_probs * advs)
    return loss


huber_loss = tf.keras.losses.Huber(reduction = tf.keras.losses.Reduction.SUM)


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


class ReinforceBaseline:
    def __init__(self, n_actions, p_lr=.001, c_lr=.001):
        self.policy = Policy(n_actions=n_actions)
        self.critic = Critic()
        self.p_optim = Adam(learning_rate=p_lr)
        self.c_optim = Adam(learning_rate=c_lr)

    @tf.function
    def train_step(self, env, initial_state: tf.Tensor, gamma: float,
                   max_steps_per_episode: int) -> tf.Tensor:
        with tf.GradientTape() as tape, tf.GradientTape() as tape2:
            action_probs, values, rewards = run_episode(env, initial_state,
                                                        gamma,
                                                        max_steps_per_episode)

            returns = get_expected_return(rewards, gamma)

            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]
            ]

          actor_loss = compute_loss(action_probs, returns)
          critic_loss = huber_loss(values,returns)

        actor_grads = tape.gradient(actor_loss,self.policy.trainable_variables)
        critic_loss = tape2.gradient(critic_loss,self.critic.trainable_variables)
        self.p_optim.apply_gradients(zip(loss,self.policy.trainable_variables))
        self.c_optim.apply_gradients(zip(loss,self.policy.trainable_variables))
        episode_reward = tf.math.reduce_sum(rewards)
        return episode_reward
