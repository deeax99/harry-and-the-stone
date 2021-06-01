import collections
import numpy as np
import tensorflow as tf
from unity import Unity
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
import os
import math
import cProfile

env = Unity()
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
num_actions = 4
num_hidden_units = 128 

class Actor(tf.keras.Model):
    def __init__(self, num_actions: int, num_hidden_units: int):
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x)


class Critic(tf.keras.Model):
    def __init__(self,  num_hidden_units: int):
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.critic(x)


class ActorCritic():
    def __init__(self, optimizer, num_actions, num_hidden_units):
        self.actor_model = Actor(num_actions, num_hidden_units)
        self.critic_model = Critic(num_hidden_units)
        self.optimizer = optimizer

    def predict(self, actor_state, fully_state):
        return self.actor_model(actor_state), self.critic_model(fully_state)

    def apply_lose(self, actor_loss, critic_loss):
        with tf.GradientTape() as tape:
            grads_actor = tape.gradient(
                actor_loss, self.actor_model.trainable_variables)
            grads_critic = tape.gradient(
                critic_loss, self.critic_model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads_actor, self.actor_model.trainable_variables))
        self.optimizer.apply_gradients(
            zip(grads_critic, self.critic_model.trainable_variables))

    def get_expected_rewards(self, reward):
        exp_reward = []
        reward_sum = sum(reward)

        for i in range(len(reward)):
            exp_reward.append(reward_sum)
            reward_sum -= reward[i]

        return exp_reward

    def advantage_compute(self, values, rewards, epsiodes):
        advantage = []

        for i in range(epsiodes):
            advantage.append(rewards[i] - values[i])

        return advantage

    def actions_log_compute(self, actions):
        action_log = []

        for action in actions:
            action_log.append(math.log(action))

        return action_log

    def compute_lose(self, action_probs, values, rewards, epsiodes):

        rewards = self.get_expected_rewards(rewards)
        advantage = self.advantage_compute(values, rewards, epsiodes)
        actions_log = self.actions_log_compute(action_probs)

        actor_lose = 0
        for i in range(epsiodes):
            actor_lose -= actions_log[i] * advantage[i]

        critic_lose = huber_loss(values, rewards)
        return actor_lose, critic_lose
    
    
    def train_step( self,initial_state: tf.Tensor, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, numEpsiodes: int, max_steps_per_episode: int):

        with tf.GradientTape() as tape:

    
            action_probs, values, rewards = self.run_episode(initial_state, model, max_steps_per_episode) 

    
            returns = self.get_expected_rewards(rewards)

    
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

    
            loss = self.compute_loss(action_probs, values, returns,numEpsiodes)

 
        grads = tape.gradient(loss, model.trainable_variables)

  
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward
    
    def run_episode(self,initial_state: tf.Tensor,  model: tf.keras.Model, max_steps: int) -> List[tf.Tensor]:
        

        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
       
            state = tf.expand_dims(state, 0)

            
            action_logits_t, value = model(state)

            
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            # Store critic values
            values = values.write(t, tf.squeeze(value))

           
            action_probs = action_probs.write(t, action_probs_t[0, action])

            
            state, reward, done = self.tf_env_step(action)
            state.set_shape(initial_state_shape)

            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards
    
    def tf_env_step(self,action: tf.Tensor) -> List[tf.Tensor]:
      return tf.numpy_function(self.env_step, [action], 
                           [tf.float32, tf.int32, tf.int32])

    def env_step(self,action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
      

        state, reward, done = env.action(action)
        return (np.asarray(state), np.array(reward, np.int32), np.array(done, np.int32))



model = ActorCritic(num_actions, num_hidden_units)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


max_episodes = 1000
max_steps_per_episode = 1000
running_reward = 0
def main():
    avarege = 0
    for i in range(max_episodes):
        initial_state = tf.convert_to_tensor(env.reset())
        episode_reward = int(train_step(initial_state, model, optimizer, i, max_steps_per_episode))

        running_reward = episode_reward*0.01 + running_reward*.99

      #t.set_description(f'Episode {i}')
      #t.set_postfix(
      #    episode_reward=episode_reward, running_reward=running_reward)

      # Show average episode reward every 10 episodes
        avarege += episode_reward
        if i % 10 == 0:
            print(f'Episode {i}: average reward: {avarege / 10}')
            avarege = 0
      #if running_reward > reward_threshold:  
      #   break

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')


cProfile.run("main()" , "stat.txt")