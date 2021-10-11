import gym
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam

from PolicyGradients.Algos.Reinforce import Reinforce, Model, compute_discounted_rewards

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def _compute_policy_loss(action_probs, rewards, state_values):
    log_action_probs = tf.math.log(action_probs)
    advs = tf.stop_gradient(rewards - state_values)
    return -tf.math.reduce_sum(log_action_probs * advs)


class ReinforceBaseLine(Reinforce):
    def __init__(self, env, actor_lr, critic_lr, policy, critic, gamma, max_episodes, max_eps_steps):
        super().__init__(env, actor_lr, policy, gamma, max_episodes, max_eps_steps)
        self.critic = critic
        self.critic_optimizer = Adam(critic_lr)

    def _run_episode(self):
        state = tf.constant(self.env.reset(), dtype=tf.float32)
        rewards = tf.TensorArray(tf.float32, 0, True)
        action_probs = tf.TensorArray(tf.float32, 0, True)
        state_values = tf.TensorArray(tf.float32, 0, True)

        state_shape = state.shape

        for step in tf.range(self.max_eps_steps):
            action, action_logits_step = self.get_action(state)
            action_probs_step = tf.nn.softmax(action_logits_step)[0, action]
            state, reward, done = self.tf_env_step(action)
            value = self.critic(tf.expand_dims(state, 0))

            self.steps_taken += 1

            action_probs = action_probs.write(step, action_probs_step)
            rewards = rewards.write(step, reward)
            state_values = state_values.write(step, value)

            state.set_shape(state_shape)

            if tf.cast(done, tf.bool):
                break
        return action_probs.stack(), rewards.stack(), state_values.stack()

    def train(self):
        with tf.GradientTape() as tape, tf.GradientTape() as tape2:
            action_probs, rewards, values = self._run_episode()
            discounted_rewards = compute_discounted_rewards(rewards, self.gamma)
            policy_loss = _compute_policy_loss(action_probs, discounted_rewards, values)
            critic_loss = huber_loss(values,discounted_rewards)
        policy_grads = tape.gradient(policy_loss, self.policy.trainable_variables)
        critic_grads = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return rewards


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    policy = Model(env.action_space.n, hidden_units=12)
    critic = Model(1, hidden_units=12)

    reinforce_baseline = ReinforceBaseLine(env, actor_lr=.0025, critic_lr=.02, critic=critic, policy=policy, gamma=.99,
                                           max_episodes=10000, max_eps_steps=250)

    running_rewards, episode_rewards = reinforce_baseline()
    episodes = np.arange(len(running_rewards))
    plt.plot(episodes, running_rewards, label='running_reward')
    plt.plot(episodes, episode_rewards, label='episode_reward')
    plt.legend()
    plt.show()

    reinforce_baseline.demo()
