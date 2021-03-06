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

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

DEBUG = False

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

num_hidden_units = 512
binary_hidden_units = 256

class Actor(tf.keras.Model):
    def __init__(self, num_actions: int):
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)

    def call(self, inputs):
        x = self.common(inputs)
        return self.actor(x)

    def actor_predict(self,inputs):
        actions_probility = self.call(inputs)
        action_probility_logits = tf.nn.softmax(actions_probility)
        actions = tf.random.categorical(actions_probility,1,dtype=tf.int32)
        return actions[0,0].numpy(),action_probility_logits[0,actions[0,0]]

class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor):
        x = self.common(inputs)
        return self.critic(x)

class ActorCritic():

    def __init__(self, num_actions):

        self.actor_model = Actor(num_actions)
        self.critic_model = Critic()
        self.clear()

    def save(self, actor_name):
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(12))
        self.actor_model.save(
            "Model/{}/ActorModel/{}".format(result_str, actor_name))
        self.critic_model.save(
            "Model/{}/CriticModel/{}".format(result_str, actor_name))

    def predict(self, actor_state, fully_state, debug=False):
        with self.actor_tape, self.critic_tape:

            actor_state = tf.expand_dims(actor_state, 0)
            fully_state = tf.expand_dims(fully_state, 0)

            action, act_prob = self.actor_model.actor_predict(actor_state)
            expected_reward = self.critic_model(fully_state)[0]

            self.actor_predict.append(act_prob)
            self.critic_predict.append(expected_reward)

            return action

    def apply_loss(self, reward):

        with self.actor_tape, self.critic_tape:
            actor_loss, critic_loss = self.compute_loss(reward)

        grads_actor, grads_critic = self.compute_gradient(
            actor_loss, critic_loss)

        optimizer.apply_gradients(
            zip(grads_actor, self.actor_model.trainable_variables))

        optimizer.apply_gradients(
            zip(grads_critic, self.critic_model.trainable_variables))

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
        expected_rewards = self.get_expected_return(reward)
        actions_log = self.actions_log_compute(self.actor_predict)
        advantage = self.advantage_compute(
            self.critic_predict, expected_rewards)

        actor_loss = -tf.reduce_sum(actions_log * advantage)
        critic_loss = huber_loss(self.critic_predict, expected_rewards)

        return actor_loss, critic_loss

    def get_expected_return(self, rewards: tf.Tensor, gamma: float = .99, standardize: bool = True) -> tf.Tensor:
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
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) /
                       (tf.math.reduce_std(returns) + eps))

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

        self.first_thieve = []
        self.second_thieve = []

        self.first_thieve_done = False
        self.second_thieve_done = False

        self.done = False


class Worker():
    def __init__(self, env, is_subprocess=True,worker_id = 0):
        self.instance = InstanceInfo(env, worker_id)

        self.harry_a2c = ActorCritic(6)
        self.first_thieve_a2c = ActorCritic(8)
        self.second_thieve_a2c = ActorCritic(8)

    def create_subprocess(self, port, is_subprocess , instance_count , continer_id = 0):
        if is_subprocess:
            app_name = 'harry_game_with_renderer' if i == 0 else 'harry_game_server'
            args = [app_name + "/HarryAndTheStone.exe",'--p', str(port),'--c',str(unity_count)]
            self.subprocess = subprocess.Popen(
                args, stdout=subprocess.DEVNULL)
            print("process {}:{} opened".format(port,continer_id))

        tcp = TCPConnection(port)
        return Unity(port,tcp_connection=tcp,continer_id = continer_id)

    def apply_harry(self, env, action, env_actions):
        x = (int(action) & 1) - 1
        y = (int(action) & 2) - 1

        env.apply_harry_action(env_actions, x, y)

    def apply_thieve(self, env, action, env_actions, thieve_id):

        x = (int(action) & 1) - 1
        y = (int(action) & 2) - 1
        grab = (int(action) & 4)

        if thieve_id == 0:
            env.apply_first_thieve_action(env_actions, x, y, grab)
        else:
            env.apply_second_thieve_action(env_actions, x, y, grab)

    def run_episode(self):
        self.instance.reset()
        while (True):
            env_actions = {}
            full_state, hr_state, ft_state, st_state = self.instance.state
            
            hr_action = self.harry_a2c.predict(hr_state,full_state)
            self.apply_harry(self.instance.env, hr_action, env_actions)

            if not self.instance.first_thieve_done:
                ft_action = self.first_thieve_a2c.predict(ft_state,full_state)            
                self.apply_thieve(self.instance.env, ft_action, env_actions,0)

            if not self.instance.second_thieve_done:
                st_action = self.second_thieve_a2c.predict(st_state,full_state)   
                self.apply_thieve(self.instance.env, st_action, env_actions,1)

            state, rewards, done = self.instance.env.action(env_actions)

            self.instance.state = state

            self.instance.harry_reward.append(rewards[0])
            
            if not self.instance.first_thieve_done:
                self.instance.first_thieve.append(rewards[1])

            if not self.instance.second_thieve_done:
                self.instance.second_thieve.append(rewards[2])

            if done[0]:
                break
            
            if done[1]:
                self.instance.first_thieve_done = True

            if done[2]:
                self.instance.second_thieve_done = True

    def train(self, iteration_count):
        print("Strat Tranning")
        t = time.time()
        for i in range(iteration_count):
            self.run_episode()
            self.harry_a2c.apply_loss(self.instance.harry_reward)
            self.first_thieve_a2c.apply_loss(self.instance.first_thieve)
            self.second_thieve_a2c.apply_loss(self.instance.second_thieve)
            print("it {} = {}".format(i,time.time() - t))

        self.close()

    def close(self):
        return
        if subprocess:
            self.subprocess.terminate()

#tf.random.set_seed(seed)
#np.random.seed(seed)
def test():

    a = Actor(8)
    #a_s = [1.4,3,5,5,34,3,1,23,3,3,2,2,2]
    #a.actor_predict(a_s)
    a_s =  [(j+3)/(j + 1) for j in range(16)] 
    a_s = tf.expand_dims(a_s,0)
    t = time.time()
    with tf.GradientTape() as tape:
        loss = tf.constant(0.0)
        print("Start")
        for i in range(1050):
            x = a.actor_predict(a_s)
            loss += tf.reduce_sum(x[1])
            print(i,time.time() - t)
        print("End")
    #grad = tape.gradient(x,a.trainable_variables)
    #print(grad)
    #print(x)

def open_unity(port=7979,visual = True):
    app_name = 'harry_game_with_renderer' if visual else 'harry_game_server'
    args = [app_name + "/HarryAndTheStone.exe",'--p', str(port),'--c',str(1)]
    subprocess.Popen(args, stdout=subprocess.DEVNULL)
    print("process {} opened".format(port))
    tcp = TCPConnection(port)
    return Unity(port,tcp_connection=tcp,continer_id = 0)


def start_worker(port=7979):
    env = open_unity(port=port, visual=False)
    worker = Worker(env, is_subprocess=True)
    worker.train(10000)


def main():
      start_worker()      

if __name__ == "__main__":
    with tf.device("/cpu:0"):
        main()
