import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import (Dense)
from tensorflow.keras import optimizers
import numpy as np
from tensorflow.keras import regularizers

class Policy(tf.keras.Model):
    def __init__(self,n_actions):
        super(Policy, self).__init__(name='Policy')
        self.fc1 = Dense(20,activation='relu')
        self.fc3 = Dense(n_actions,activation='softmax')

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        action_probs = self.fc3(x)
        return action_probs


def loss(probs,action,reward):
    distr = tfp.distributions.Categorical(probs)
    return -distr.log_prob(action) * reward


def calc_discounted_returns(rewards,gamma):
    discounted_rewards = np.zeros(len(rewards))
    rewards.reverse()
    for t in range(len(rewards)):
        discounted_rewards[t] = rewards[t] if t < 1 else (rewards[t] + discounted_rewards[t-1])
    gammas = [gamma ** t for t in range(len(rewards))]
    return discounted_rewards[::-1] * gammas



class Reinforce:
    def __init__(self,n_actions,gamma = .99, lr=.001):
        self.gamma = gamma
        self.policy = Policy(n_actions=n_actions)
        self.optim = optimizers.RMSprop(learning_rate=lr)

    def get_action(self,state):
        probs = self.policy(state[None])
        distr = tfp.distributions.Categorical(probs)
        return distr.sample().numpy()[0]

    def save(self):
        self.policy.save('policy')
    def load(self):
        self.policy = tf.keras.models.load_model('policy')

    def train_iter(self,states,actions,rewards):
        discounted_returns = calc_discounted_returns(rewards,self.gamma)
        with tf.GradientTape() as tape:
            probs = self.policy(tf.constant(states))
            l = tf.reduce_sum(loss(probs,actions,discounted_returns))
        gradients = tape.gradient(l,self.policy.trainable_variables)
        self.optim.apply_gradients(zip(gradients,self.policy.trainable_variables))