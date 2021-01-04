import tensorflow as tf
from tensorflow.keras.layers import (Dense,InputLayer)
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
import gym
import numpy as np

class PolicyNetwork(tf.keras.Model):

	def __init__(self,env):
		super(PolicyNetwork,self).__init__(name="PolicyNetwork")
		self.input_layer = InputLayer(input_shape=env.observation_space.shape)
		self.fc1 = Dense(32,activation='relu')
		self.fc2 = Dense(32,activation='relu')
		self.output_layer = Dense(env.action_space.n,activation='softmax')
	
	@tf.function
	def call(self,inputs):
		x = self.input_layer(inputs)
		x = self.fc1(x)
		action_prefs = self.fc2(x)
		action_probs = self.output_layer(action_prefs)
		return action_probs

def reinforce(env,policy_gradient,num_episodes=1000,gamma=.99):	
	optim = tf.keras.optimizers.Adam()

	def loss(p,action,reward):
		distr = tfp.distributions.Categorical(probs=p)
		# use negative because optimizers package implements gradient descent not ascent
		return -distr.log_prob(action) * reward

	def discount_rewards(rewards):
		sum_reward = 0
		discount_rewards = []
		rewards.reverse()
		for r in rewards:
			sum_reward = r + gamma * sum_reward
			discount_rewards.append(sum_reward)
		discount_rewards.reverse()
		return discount_rewards

	def update_network(rewards,actions,states):
		discounted_rewards = discount_rewards(rewards)
		for [state,action,reward] in zip(states,actions,discounted_rewards):
			with tf.GradientTape() as tape:
				probs = policy_gradient(state[None])
				l = loss(probs,action,reward)
			gradients = tape.gradient(l,policy_gradient.trainable_variables)
			optim.apply_gradients(zip(gradients,policy_gradient.trainable_variables))

	def act(s):
		probs = policy_gradient(s[None])
		distr = tfp.distributions.Categorical(probs=probs)
		a = distr.sample()
		return int(a.numpy()[0])


	episode_rewards,episode_means = [],[]

	for episode in range(num_episodes):
		rewards,states,actions,episode_reward = [],[],[],0
		state = env.reset()
		
		while True:
			action = act(state)
			next_state,reward,done,_ = env.step(action)
			rewards.append(reward)
			states.append(state)
			actions.append(action)

			episode_reward += reward

			if done:
				episode_rewards.append(episode_reward)
				mean = np.mean(episode_rewards[-100:])
				episode_means.append(mean)
				if mean >= 200:
					return episode_rewards,episode_means
				print(f"episode: {episode},episode_reward:{episode_reward},episode_mean:{mean}")
				update_network(rewards,actions,states)
				break
			else:
				state = next_state

	return episode_rewards,episode_means

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	policy_gradient = PolicyNetwork(env)

	episode_rewards,mean_rewards = reinforce(env,policy_gradient)


	plt.plot(episode_rewards)
	plt.plot(mean_rewards)


