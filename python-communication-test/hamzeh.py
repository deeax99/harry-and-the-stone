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
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
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
  #print(prob,"prob")
  action = tf.random.categorical(prob, 1)[0, 0]
  return action , prob[0 , action]

  
def train(probs, rewards, actions):
  sum_reward = 0
  discnt_rewards = []
  rewards.reverse()
  for r in rewards:
    sum_reward = r + gamma*sum_reward
    discnt_rewards.append(sum_reward)
  discnt_rewards.reverse()  

  for prob, reward, action in zip(probs, discnt_rewards, actions):
    with tf.GradientTape() as tape:
      #p = model(np.array([state]), training=True)
      loss = a_loss(prob,action,reward)
      grads = tape.gradient(loss, model.trainable_variables)
      opt.apply_gradients(zip(grads, model.trainable_variables))



  
def a_loss(prob, action, reward):
  #dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
  log_prob = tf.math.log( prob )
  loss = log_prob*reward
  return loss 


unity = Unity()

s=0
while(True):
  
  done = False
  state = unity.reset()
  total_reward = 0
  rewards = []
  probs = []
  actions = []
  while not done:
    #env.render()
    action , prob =act(state)
    #print(action)
    next_state,reward,done= unity.action(action)
    
    rewards.append(reward)
    
    probs.append(prob)
    
    actions.append(action)
    
    state = next_state
    
    total_reward += reward
    
    if done:
      train(probs, rewards, actions)
      #print(actions,"here")
      s+=1
      #print("total step for this episord are {}".format(t))
      print("total reward = {} episd = {}".format(total_reward,s))