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
num_actions = 2
num_hidden_units = 128 

class Actor(tf.keras.Model):
    def __init__(self, num_actions: int, num_hidden_units: int):
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)

    def call(self, inputs) :
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

    def advantage_compute(self, values, rewards):
        advantage = []
        framecount=len(rewards)
        for i in range(framecount):
            advantage.append(rewards[i] - values[i])

        return advantage

    def actions_log_compute(self, actions):
        action_log = []

        for action in actions:
            action_log.append(math.log(action))

        return action_log

    def compute_lose(self, action_probs, values, rewards ):
        framecount=len(rewards)
        rewards = self.get_expected_rewards(rewards)
        advantage = self.advantage_compute(values, rewards, framecount)
        actions_log = self.actions_log_compute(action_probs)

        actor_lose = 0
        for i in range(framecount):
            actor_lose -= actions_log[i] * advantage[i]

        critic_lose = huber_loss(values, rewards)
        return actor_lose, critic_lose
    


def train(first_state,max_steps):
    harry_act_probs, first_thief_act_probs, second_thief_act_probs,harry_vals,first_thief_vals,second_thief_vals,harry_rewards,first_thief_rewards,second_thief_rewards= start_ep(first_state,max_steps)

    harry_expected_rewards=harry.get_expected_rewards(harry_rewards)
    first_thief_expected_rewards=first_thief.get_expected_rewards(first_thief_rewards)
    second_thief_expected_rewards=second_thief.get_expected_rewards(second_thief_rewards)

    harry_actor_loss, harry_critic_lose = harry.compute_loss(harry_act_probs, harry_vals, harry_expected_rewards)
    first_thief_actor_loss, first_thief_critic_lose = first_thief.compute_loss(first_thief_act_probs, first_thief_vals, first_thief_expected_rewards)
    second_thief_actor_loss, second_thief_critic_lose = second_thief.compute_loss(second_thief_act_probs, second_thief_vals, second_thief_expected_rewards)

    harry.apply_lose(harry_actor_loss,harry_critic_lose)
    first_thief.apply_lose(first_thief_actor_loss,first_thief_critic_lose)
    second_thief.apply_lose(second_thief_actor_loss,second_thief_critic_lose)

    harry_episode_reward=0
    first_thief_episode_reward=0
    second_thief_episode_reward=0

    for i in harry_rewards :
        harry_episode_reward+=i
    
    for i in first_thief_rewards :
       first_thief_episode_reward+=i
    
    for i in second_thief_rewards :
        second_thief_episode_reward+=i
    
    return harry_episode_reward , first_thief_episode_reward , second_thief_episode_reward

def start_ep(initial_state , max_steps):
    harry_act_probs=[]
    first_thief_act_probs=[]
    second_thief_act_probs=[]
    
    harry_vals=[]
    first_thief_vals=[]
    second_thief_vals=[]

    harry_rewards=[]
    first_thief_rewards=[]
    second_thief_rewards=[]

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in range(max_steps):
        
        harry_act_logits_t ,harry_val=harry(state)
        first_thief_act_logits_t ,first_thief_val=first_thief(state)
        second_thief_act_logits_t ,second_thief_val=second_thief(state)

        harry_action=tf.random.categorical(harry_act_logits_t, 1)[0, 0]  #i didn't know if you want me to not use tf so i just used tf
        first_thief_action=tf.random.categorical(first_thief_act_logits_t, 1)[0, 0]
        second_thief_action=tf.random.categorical(second_thief_act_logits_t, 1)[0, 0]

        harry_action_probs_t = tf.nn.softmax(harry_act_logits_t)
        firt_thief_action_probs_t = tf.nn.softmax(first_thief_act_logits_t)
        second_thief_action_probs_t = tf.nn.softmax(second_thief_act_logits_t)

        harry_vals.append(t, tf.squeeze(harry_val) )
        first_thief_vals.append(t, tf.squeeze(first_thief_val) )
        second_thief_vals.append(t, tf.squeeze(second_thief_val) )

        harry_act_probs.append(t,harry_action_probs_t[0, harry_action])
        first_thief_act_probs.append(t,firt_thief_action_probs_t[0, first_thief_action])
        second_thief_act_probs.append(t,second_thief_action_probs_t[0, second_thief_action])

        state, harry_reward, harry_done = tf_env_step(harry_action)
        state, first_thief_reward, first_thief_done = tf_env_step(first_thief_action)
        state, second_thief_reward, second_thief_done = tf_env_step(second_thief_action)

        state.set_shape(initial_state_shape)

        harry_rewards.append(harry_reward)
        first_thief_rewards.append(first_thief_reward)
        second_thief_rewards.append(second_thief_rewards)

        if harry_done and first_thief_done and second_thief_done :
            break
    

    return harry_act_probs, first_thief_act_probs, second_thief_act_probs,harry_vals,first_thief_vals,second_thief_vals,harry_rewards,first_thief_rewards,second_thief_rewards

   
def tf_env_step(self,action) :
    return tf.numpy_function(self.env_step, [action], [tf.float32, tf.int32, tf.int32])

def env_step(self,action):
    state, reward, done = env.action(action)
    return (np.asarray(state), np.array(reward, np.int32), np.array(done, np.int32))



harry = ActorCritic(num_actions, num_hidden_units)

first_thief =ActorCritic(3, num_hidden_units)

second_thief = ActorCritic(3, num_hidden_units)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


max_episodes = 1000
max_steps_per_episode = 1000
running_reward = 0

def main():
    avarege = 0
    for i in range(max_episodes):
        initial_state = tf.convert_to_tensor(env.reset())
        episode_reward = int(train(initial_state, max_steps_per_episode))
        running_reward = episode_reward*0.01 + running_reward*.99

     
        avarege += episode_reward
        if i % 10 == 0:
            print(f'Episode {i}: average reward: {avarege / 10}')
            avarege = 0
      #if running_reward > reward_threshold:  
      #   break

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')


cProfile.run("main()" , "stat.txt")















"""

def train_step(initial_state, model, optimizer, max_steps_per_episode):
        
       

    
    action_probs, values, rewards = run_episode(initial_state, model, max_steps_per_episode) 

    
    returns = model.get_expected_rewards(rewards)

    
    action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

    
    actor_loss, critic_lose = model.compute_loss(action_probs, values, returns)

 
    model.apply_lose(actor_loss,critic_lose)

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward
    

"""

"""


def run_episode(initial_state, max_steps):
        

    action_probs = []
    values = [] 
    rewards = []

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in range(max_steps):
       
        state = tf.expand_dims(state, 0)
            
        action_logits_t, value = harry(state)

            
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

            # Store critic values
        values = values.write(t, tf.squeeze(value))

           
        action_probs = action_probs.write(t, action_probs_t[0, action])

            
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards
    """