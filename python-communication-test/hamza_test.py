import socket
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from unity import Unity
import tensorflow_probability as tfp

num_inputs = 4
num_actions = 4
num_hidden = 128
gamma=0.99
opt = keras.optimizers.Adam(learning_rate=0.01)
class Model(tf.keras.Model):
  
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(30,activation='relu')
    self.d2 = tf.keras.layers.Dense(30,activation='relu')
    self.out = tf.keras.layers.Dense(num_actions,activation='softmax')

  def call(self, input_data):
    x = tf.convert_to_tensor(input_data)
    x = self.d1(x)
    x = self.d2(x)
    x = self.out(x)
    return x


model = Model()


def act(state):
  prob = model(np.array([state]))
  dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
  action = dist.sample()
  return int(action.numpy()[0])

  
def train(states, rewards, actions):
  sum_reward = 0
  discnt_rewards = []
  rewards.reverse()
  for r in rewards:
    sum_reward = r + gamma*sum_reward
    discnt_rewards.append(sum_reward)
  discnt_rewards.reverse()  

  for state, reward, action in zip(states, discnt_rewards, actions):
    with tf.GradientTape() as tape:
      p = model(np.array([state]), training=True)
      loss = a_loss(p,action,reward)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))



  
def a_loss(prob, action, reward): 
  dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
  log_prob = dist.log_prob(action)
  loss = -log_prob*reward
  return loss 


unity = Unity()

steps = 200
for s in range(steps):
  
  done = False
  state = unity.reset()
  total_reward = 0
  rewards = []
  states = []
  actions = []
  while not done:
    #env.render()
    action =act(state)
    #print(action)
    next_state,reward,done= unity.action(action)
    
    rewards.append(reward)
    
    states.append(state)
    
    actions.append(action)
    
    state = next_state
    
    total_reward += reward
    
    if done:
      train(states, rewards, actions)
      #print("total step for this episord are {}".format(t))
      print("total reward = {} episd = {}".format(total_reward,s))