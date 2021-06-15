from datetime import datetime
import random
import threading
import subprocess
import sys
import signal
import cProfile
import math
from typing import Any, List, Sequence, Tuple
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from unity import Unity
import tensorflow as tf
import collections
import numpy as np
import os
import time
from tcp import TCPUitility
from tcp import TCPConnection
import string
from tensorflow import keras

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

DEBUG = False

optimizer_actor = tf.keras.optimizers.Adam(learning_rate=3e-3)
optimizer_critic = tf.keras.optimizers.Adam(learning_rate=1e-4)

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

num_hidden_units = 32

class Actor(tf.keras.Model):
    def __init__(self, num_actions: int,num_inp):
        super().__init__()

        self.layer1 = layers.Dense(num_hidden_units, activation="sigmoid")
        self.layer2 = layers.Dense(num_hidden_units, activation="sigmoid")
        self.layer3 = layers.Dense(num_hidden_units, activation="sigmoid")
        self.actor = layers.Dense(num_actions, activation="sigmoid")
        s = tf.expand_dims([0.0]*num_inp,0)
        self.call(s)
        self.predict(s)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(inputs)
        x = self.layer3(inputs)

        return self.actor(x)

class Critic(tf.keras.Model):
    def __init__(self,num_inp):
        super().__init__()

        self.layer1 = layers.Dense(num_hidden_units, activation="tanh")
        self.layer2 = layers.Dense(num_hidden_units, activation="tanh")
        self.layer3 = layers.Dense(num_hidden_units, activation="tanh")
        self.critic = layers.Dense(1 , activation="tanh")
        s = tf.expand_dims([0.0]*num_inp,0)
        self.call(s)
        self.predict(s)
        

    def call(self, inputs: tf.Tensor):
        x = self.layer1(inputs)
        x = self.layer2(inputs)
        x = self.layer3(inputs)

        return self.critic(x)

class ActorCritic():

    def __init__(self, num_actions,act_inp,crt_inp):

        self.actor_model = Actor(num_actions,act_inp)
        self.critic_model = Critic(crt_inp)
        self.clear()
    
    def load(self,actor_name,result_str):
        self.actor_model = keras.models.load_model('Model/{}/{}/ActorModel'.format(result_str, actor_name) , custom_objects={"Actor": Actor})
        self.critic_model =keras.models.load_model("Model/{}/{}/CriticModel".format(result_str, actor_name),custom_objects={"Critic": Critic} )
        


    def save(self, actor_name,result_str):
        
        
        self.actor_model.save(
            "Model/{}/{}/ActorModel".format(result_str, actor_name))
        self.critic_model.save(
            "Model/{}/{}/CriticModel".format(result_str, actor_name))

    def actor_predict_f(self,actor,inputs):
        actions_probility = actor.call(inputs)
        action_probility_logits = tf.nn.softmax(actions_probility)
        actions = tf.random.categorical(actions_probility,1,dtype=tf.int32)
        return actions[0,0].numpy(),action_probility_logits[0,actions[0,0]]

    def predict(self, actor_state, fully_state, debug=False):
        with self.actor_tape, self.critic_tape:

            actor_state = tf.expand_dims(actor_state, 0)
            fully_state = tf.expand_dims(fully_state, 0)

            action, act_prob = self.actor_predict_f(self.actor_model,actor_state)
            expected_reward = self.critic_model(fully_state)[0]

            self.actor_predict.append([act_prob])
            self.critic_predict.append(expected_reward)

            return action

    def apply_loss(self, reward):

        with self.actor_tape, self.critic_tape:
            actor_loss, critic_loss = self.compute_loss(reward)
        print("actor loss {} , critic loss {}".format(actor_loss,critic_loss))
        grads_actor, grads_critic = self.compute_gradient(
            actor_loss, critic_loss)

        optimizer_actor.apply_gradients((grad, var) for (grad, var) in zip(
            grads_actor, self.actor_model.trainable_variables) if grad is not None)

        optimizer_critic.apply_gradients((grad, var) for (grad, var) in zip(
            grads_critic, self.critic_model.trainable_variables) if grad is not None)

        """
        optimizer.apply_gradients(
            zip(grads_actor, self.actor_model.trainable_variables))

        optimizer.apply_gradients(
            zip(grads_critic, self.critic_model.trainable_variables))
        """
        self.clear()

    def compute_gradient(self, actor_loss, critic_loss):
        grads_actor = self.actor_tape.gradient(
            actor_loss, self.actor_model.trainable_variables)

        grads_critic = self.critic_tape.gradient(
            critic_loss, self.critic_model.trainable_variables)

        return grads_actor, grads_critic

    def compute_loss(self, reward):

        self.actor_predict = tf.convert_to_tensor(self.actor_predict)
        self.critic_predict = tf.convert_to_tensor(self.critic_predict)

        reward = tf.convert_to_tensor(reward)
        expected_rewards = self.get_expected_return(reward,standardize = True)
        actions_log = self.actions_log_compute(self.actor_predict)
        advantage = self.advantage_compute(
            self.critic_predict, expected_rewards)

        actor_loss = -tf.reduce_sum(actions_log * advantage)
        critic_loss = huber_loss(self.critic_predict, expected_rewards)

        return actor_loss , critic_loss

    def get_expected_return(self, rewards: tf.Tensor, gamma: float = .9999, standardize: bool = True) -> tf.Tensor:
        """Compute expected returns per timestep."""
        eps = np.finfo(np.float32).eps.item()

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, [discounted_sum])
        returns = returns.stack()[::-1]

        if standardize:
            returns /= 10
            #returns = ((returns - tf.math.reduce_mean(returns)) /
            #           (tf.math.reduce_std(returns) + eps))

        return returns

    def advantage_compute(self, values, rewards):
        return rewards - values

    def actions_log_compute(self, actions):
        return tf.math.log(actions)

    def get_action(self, action_distribution):
        return tf.random.categorical(action_distribution, 1).numpy()

    def clear(self):
        self.actor_tape = tf.GradientTape()
        self.critic_tape = tf.GradientTape()
        with self.actor_tape,self.critic_tape:
            self.actor_predict = []
            self.critic_predict = []


class InstanceInfo():
    def __init__(self, env:Unity, instance_id):
        self.instance_id = instance_id
        self.env = env
        self.reset()

    def reset(self):
        self.state = self.env.reset()
        self.harry_reward = []
        self.done = False


class Worker():
    def __init__(self, env, is_subprocess=True,worker_id = 0):
        self.instance = InstanceInfo(env, worker_id)
        self.harry_a2c = ActorCritic(4,4,4)

    def run_episode(self):
        self.instance.reset()
        while (True):
            env_actions = {}
            state = self.instance.state
            
            action = self.harry_a2c.predict(state,state)

            state, reward, done = self.instance.env.action(action)

            self.instance.state = state

            self.instance.harry_reward.append(reward)
            
            if done:
              break
            
        return sum(self.instance.harry_reward)

    def train(self, iteration_count):
        print("Strat Tranning")
        t = time.time()
        for i in range(iteration_count):
            hr_r = self.run_episode()
            self.harry_a2c.apply_loss(self.instance.harry_reward)
            print("it {} = harry = {} time = {}".format(
                i, hr_r, time.time() - t))
            if i %1000==0:
                letters = string.ascii_lowercase
                result_str = ''.join(random.choice(letters) for i in range(12))
                result_str=result_str + "_"+ str(i)
                self.harry_a2c.save("harry",result_str)

        self.close()

    def close(self):
        return
        if subprocess:
            self.subprocess.terminate()

app_name = 'Environment/HarryAndTheStone.exe' 
args = [app_name ]
subprocess.Popen(args, stdout=subprocess.DEVNULL)

unity = Unity()


def main ():
  worker = Worker(unity)
  worker.train(10000)
  
if __name__ == "__main__":
  main()

