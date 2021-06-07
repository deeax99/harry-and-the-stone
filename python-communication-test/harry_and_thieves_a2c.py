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

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

DEBUG = True

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

num_hidden_units = 512


class Actor(tf.keras.Model):
    def __init__(self, num_actions: int):
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)

    def call(self, inputs):
        x = self.common(inputs)
        return self.actor(x)


class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor):
        x = self.common(inputs)
        return self.critic(x)


class ActorCritic():
    def __init__(self, num_actions, actor_input_size, critic_input_size, instance_count, actions_partition):

        self.actor_model = Actor(num_actions)
        self.critic_model = Critic()

        actor_input_shape = [0.0] * actor_input_size
        critic_input_shape = [0.0] * critic_input_size

        self.actor_model(tf.expand_dims(actor_input_shape, 0))
        self.critic_model(tf.expand_dims(critic_input_shape, 0))

        self.actions_partition = actions_partition
        self.instance_count = instance_count

        self.clear()

    def predict(self, actor_states, fully_states, instances_id, debug=False):
        with self.actor_tape, self.critic_tape:

            actor_states = tf.expand_dims(actor_states, 0)
            fully_states = tf.expand_dims(fully_states, 0)

            batch_count = len(instances_id)

            if batch_count == 0:
                return []

            actions = [[] for i in range(batch_count)]

            probility_products = []

            actors_predicts = self.actor_model(actor_states)[0]
            critic_predicts = tf.convert_to_tensor(
                self.critic_model(fully_states))[0]
            for i in range(batch_count):

                partition_index = 0
                probility_product = tf.convert_to_tensor(1.0)

                for partition_size in self.actions_partition:

                    action_probility = actors_predicts[i][partition_index:partition_index+partition_size]
                    action_probility_logits = tf.nn.softmax(action_probility)

                    action_index = (tf.random.categorical(
                        [action_probility], 1).numpy()[0, 0])

                    actions[i].append(action_index)
                    probility_product = probility_product * \
                        action_probility_logits[action_index]

                    partition_index += partition_size

                probility_products.append(probility_product)

            for i in range(batch_count):
                instance_id = instances_id[i]
                self.critic_predicts[instance_id].append(critic_predicts[i, 0])
                self.actor_predicts[instance_id].append(probility_products[i])

        return actions

    def apply_loss(self, rewards):

        with self.actor_tape, self.critic_tape:
            actor_loss, critic_loss = self.compute_loss(rewards)

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

    def compute_loss(self, instance_reward):

        actor_loss = []
        critic_loss = []

        for i in range(len(instance_reward)):
            rewards = instance_reward[i]

            expected_rewards = self.get_expected_return(rewards)

            advantage = self.advantage_compute(
                self.critic_predicts[i], expected_rewards)

            actions_log = self.actions_log_compute(self.actor_predicts[i])
            actor_loss.append(-tf.reduce_sum(actions_log * advantage))

            critic_loss.append(huber_loss(
                self.critic_predicts[i], expected_rewards))

        return actor_loss, critic_loss

    def get_expected_rewards(self, reward):
        exp_reward = []
        reward_sum = sum(reward)

        for i in range(len(reward)):
            exp_reward.append(reward_sum)
            reward_sum -= reward[i]

        return exp_reward

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
        advantage = []
        framecount = len(values)
        for i in range(framecount):
            advantage.append(rewards[i] - values[i])

        return advantage

    def actions_log_compute(self, actions):
        return tf.math.log(actions)

    def get_action(self, action_distribution):
        return tf.random.categorical(action_distribution, 1).numpy()

    def clear(self):
        self.actor_tape = tf.GradientTape()
        self.critic_tape = tf.GradientTape()
        with self.actor_tape:
            with self.critic_tape:
                self.actor_predicts = [[] for i in range(self.instance_count)]
                self.critic_predicts = [[] for i in range(self.instance_count)]


class InstanceInfo():
    def __init__(self, env, instance_id):
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
    def __init__(self, unity_instance , unity_count, port=7979, is_subprocess=True):
        
        instance_count = unity_instance * unity_count
        
        self.unity_count = unity_count 
        self.unity_instance = unity_instance

        self.envs = []
        self.instance_count = instance_count
        self.create_subprocess(unity_instance , unity_count, port, is_subprocess)
        self.instances_info = [InstanceInfo(
            self.envs[i], i) for i in range(instance_count)]

        self.harry_a2c = ActorCritic(6, 15, 13, instance_count, [3, 3])

        self.first_thieve_a2c = ActorCritic(
            8, 14, 13, instance_count, [3, 3, 2])
        self.second_thieve_a2c = ActorCritic(
            8, 14, 13, instance_count, [3, 3, 2])

    def create_subprocess(self, unity_instance , unity_count, port, is_subprocess):
        for i in range(unity_instance):
            if is_subprocess:
                app_name = 'harry_game_with_renderer' if i == 0 else 'harry_game_server'
                args = [app_name + "/HarryAndTheStone.exe",'--p', str(port + i),'--c',str(unity_count)]
                self.subprocess = subprocess.Popen(
                    args, stdout=subprocess.DEVNULL)
                print("process {} opened".format(i + 1))

            tcp = TCPConnection(port + i)
            for j in range(unity_count):
                self.envs.append(Unity(port + i,tcp_connection=tcp,continer_id=j))

    def apply_harry(self, env, actions, env_actions):
        x = int(actions[0]) - 1
        y = int(actions[1]) - 1

        env.apply_harry_action(env_actions, x, y)

    def apply_thieve(self, env, actions, env_actions, thieve_id):

        x = int(actions[0]) - 1
        y = int(actions[1]) - 1

        grab = int(actions[2])

        if thieve_id == 0:
            env.apply_first_thieve_action(env_actions, x, y, grab)
        else:
            env.apply_second_thieve_action(env_actions, x, y, grab)

    def run_episode(self):

        remining_instance = self.instance_count

        for i in range(remining_instance):
            self.instances_info[i].reset()
        if DEBUG:
            print("start eps")
        iteration = 0
        while(remining_instance != 0):
            remining_instance = 0
            for i in range(self.instance_count):
                if not self.instances_info[i].done:
                     remining_instance+=1

            if remining_instance == 0:
                break

            fully_state = []
            harry_states = []

            first_thieve_states = []
            second_thieve_states = []

            instances_id_harry = []
            instances_id_first_thieve = []
            instances_id_second_thieve = []

            for instance_info in self.instances_info:
                if not instance_info.done:

                    full_state, harry_state, fir_thieve_state, sec_thieve_state = instance_info.state

                    fully_state.append(full_state)

                    harry_states.append(harry_state)
                    instances_id_harry.append(instance_info.instance_id)

                    if not instance_info.first_thieve_done:
                        first_thieve_states.append(fir_thieve_state)
                        instances_id_first_thieve.append(
                            instance_info.instance_id)

                    if not instance_info.second_thieve_done:
                        second_thieve_states.append(fir_thieve_state)
                        instances_id_second_thieve.append(
                            instance_info.instance_id)
            
            if DEBUG:
                print("get states")

            def harry_thread():
                self.harry_actions = self.harry_a2c.predict(
                    harry_states, fully_state, instances_id_harry)

            def ft_thread():
                self.first_thieve_actions = self.first_thieve_a2c.predict(
                    first_thieve_states, fully_state, instances_id_first_thieve)
            def st_thread():
                self.second_thieve_actions = self.second_thieve_a2c.predict(
                    second_thieve_states, fully_state, instances_id_second_thieve)
            
            h_thread = threading.Thread(target=harry_thread)
            ft_thread = threading.Thread(target=ft_thread)
            st_thread = threading.Thread(target=st_thread)

            h_thread.start()
            ft_thread.start()
            st_thread.start()

            h_thread.join()
            ft_thread.join()
            st_thread.join()

            if DEBUG:
                print("get actions")

            ft_index = 0
            st_index = 0
            envs_actions = []
            for i in range(remining_instance):
                instance = instances_id_harry[i]

                env_actions = {}
                env = self.envs[instance]

                self.apply_harry(env, self.harry_actions[i], env_actions)

                if not self.instances_info[instance].first_thieve_done:
                    self.apply_thieve(
                        env, self.first_thieve_actions[ft_index], env_actions, 0)
                    ft_index += 1

                if not self.instances_info[instance].second_thieve_done:
                    self.apply_thieve(
                        env, self.second_thieve_actions[st_index], env_actions, 1)
                    st_index += 1

                envs_actions.append(env_actions)
            
            def insert(i):   
                instance = instances_id_harry[i]
                env_actions = envs_actions[i]
                env = self.envs[instance]

                state, rewards, done = env.action(env_actions)
                self.instances_info[instance].harry_reward.append(rewards[0])

                if not self.instances_info[instance].first_thieve_done:
                    self.instances_info[instance].first_thieve.append(
                        rewards[1])

                if not self.instances_info[instance].second_thieve_done:
                    self.instances_info[instance].second_thieve.append(
                        rewards[2])

                if done[0]:
                    self.instances_info[instance].done = True
                    if DEBUG:
                        print("instance {} done".format(instance))

                if done[1]:
                    self.instances_info[instance].first_thieve_done = True

                if done[2]:
                    self.instances_info[instance].second_thieve_done = True
                
                

            threads = [threading.Thread(target=insert,args=(i,)) for i in range(remining_instance)]
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            del threads
            if DEBUG:
                print("action applied {}".format(iteration))
            iteration += 1


    def train(self, max_ep):
        print("Strat Tranning")
        batch_index = 0
        for i in range(max_ep):
            self.run_episode()

            harry_rewards = []
            ft_rewards = []
            st_rewards = []

            #harry_sum = sum(harry_rewards[0]) / len(harry_rewards[0])
            #ft_rewards = sum(ft_rewards[0]) / len(ft_rewards[0])
            #st_rewards = sum(st_rewards[0]) / len(harry_rewards[0])


            for info in self.instances_info:
                harry_rewards.append(info.harry_reward)
                ft_rewards.append(info.first_thieve)
                st_rewards.append(info.second_thieve)
            
            if DEBUG:
                print("start apply loss")

            harry_loss = threading.Thread(target=self.harry_a2c.apply_loss,args=(harry_rewards,))
            ft_loss = threading.Thread(target=self.first_thieve_a2c.apply_loss,args=(ft_rewards,))
            st_loss = threading.Thread(target=self.second_thieve_a2c.apply_loss,args=(st_rewards,))
            
            harry_loss.start()
            ft_loss.start()
            st_loss.start()

            harry_loss.join()
            ft_loss.join()
            st_loss.join()
            
            print(str(batch_index) + " : " +
                  str(batch_index + self.instance_count))
                  
            print("Harry reward")
            batch_index += self.instance_count

        self.close()

    def close(self):
        return
        if subprocess:
            self.subprocess.terminate()

#tf.random.set_seed(seed)
#np.random.seed(seed)


def main():
    worker = Worker(unity_instance=5,unity_count=3, port=7979, is_subprocess=True)
    worker.train(10000)

if __name__ == "__main__":
    main()
