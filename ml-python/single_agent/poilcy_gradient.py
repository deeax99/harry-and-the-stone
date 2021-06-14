import socket
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from unity import Unity
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import subprocess

all_ep_reward=[]
ep_count=0

num_inputs = 4
num_actions = 4
num_hidden = 128
gamma=0.99


opt = keras.optimizers.Adam(learning_rate=0.01)

app_name = 'Environment/HarryAndTheStone.exe' 
args = [app_name ]
subprocess.Popen(args, stdout=subprocess.DEVNULL)


unity = Unity()

class poilcy_gradient(tf.keras.Model):
  
  def __init__(self):
    super().__init__()
    self.layer1 = tf.keras.layers.Dense(30,activation='relu')
    self.layer2 = tf.keras.layers.Dense(30,activation='relu')
    self.out = tf.keras.layers.Dense(num_actions,activation='softmax')

  def call(self, input_data):
    layer_Output = tf.convert_to_tensor(input_data)
    layer_Output = self.layer1(layer_Output)
    layer_Output = self.layer2(layer_Output)
    layer_Output = self.out(layer_Output)
    return layer_Output

model = poilcy_gradient()

def Chose_Action(state):
  prob = model(np.array([state]))
  dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
  action = dist.sample()
  return int(action.numpy()[0])
  
def learn(states, rewards, actions):
  reward_sum = 0
  dis_rewards = []
  rewards.reverse()

  for r in rewards:
    reward_sum = r + gamma*reward_sum
    dis_rewards.append(reward_sum)

  dis_rewards.reverse()

  for state, reward, action in zip(states, dis_rewards, actions):
    with tf.GradientTape() as tape:
      p = model(np.array([state]), training=True)
      loss = calculate_loss(p,action,reward)
      gradient = tape.gradient(loss, model.trainable_variables)
      opt.apply_gradients(zip(gradient, model.trainable_variables))

  
def calculate_loss(prob, action, reward): 
  dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
  log_prob = dist.log_prob(action)
  loss = -log_prob * reward
  return loss 


while(True):
  done = False
  state = unity.reset()
  total_reward = 0
  rewards = []
  states = []
  actions = []
  while not done:
    
    action =Chose_Action(state)
    
    next_state,reward,done= unity.action(action)
    
    rewards.append(reward)
    
    states.append(state)
    
    actions.append(action)
    
    state = next_state
    
    total_reward += reward
    
    if done:
          all_ep_reward.append(total_reward)
          ep_count+=1

          if ep_count%10000==0 :
                
                plt.plot(np.arange(ep_count),all_ep_reward , color='green')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title('Reward Per Episode Using Q-Learning  ', y=1.1)
                plt.show()

          learn(states, rewards, actions)
          print("total reward = {} episd = {}".format(total_reward,ep_count))