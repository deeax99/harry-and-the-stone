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
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

"""
def handler(signum, frame):
    print("Bye")
    sys.exit(0)
signal.signal(signal.SIGINT, handler)
"""

eps_count = 0


class Actor(tf.keras.Model):
    def __init__(self, num_actions: int, num_hidden_units: int):
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)

    def call(self, inputs):
        x = self.common(inputs)
        return self.actor(x)


class Critic(tf.keras.Model):
    def __init__(self,  num_hidden_units: int):
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor):
        x = self.common(inputs)
        return self.critic(x)


class ActorCritic():
    def __init__(self, optimizer, num_actions, num_hidden_units, actions_range):

        self.actor_model = Actor(num_actions, num_hidden_units)
        self.critic_model = Critic(num_hidden_units)

        self.optimizer = optimizer
        self.huber_loss = tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.SUM)

        self.actions_range = actions_range

        self.clear()

    def predict(self, actor_state, fully_state, debug=False):

        actor_state = tf.expand_dims(actor_state, 0)
        fully_state = tf.expand_dims(fully_state, 0)

        with self.actor_tape:
            with self.critic_tape:

                actor_predict = self.actor_model(actor_state)[0]

                actions = []

                prod = tf.convert_to_tensor(1.0)

                start_index = 0
                for action in self.actions_range:
                    action_probility = actor_predict[start_index:start_index+action]
                    action_probility_logits = tf.nn.softmax(action_probility)
                    start_index += action

                    actions.append(self.get_action(
                        tf.expand_dims(action_probility, 0))[0, 0])
                    prod = prod * action_probility_logits[actions[-1]]

                critic_predict = self.critic_model(fully_state)

                self.actor_predicts.append(prod)
                self.critic_predicts.append(critic_predict[0, 0])
                if debug:
                    print(prod, critic_predict[0, 0])

                return actions, critic_predict[0, 0]

    def apply_loss(self, rewards):

        actor_loss, critic_loss = self.compute_loss(rewards)

        grads_actor, grads_critic = self.compute_gradient(
            actor_loss, critic_loss)

        self.optimizer.apply_gradients(
            zip(grads_actor, self.actor_model.trainable_variables))

        self.optimizer.apply_gradients(
            zip(grads_critic, self.critic_model.trainable_variables))

        self.clear()

    def compute_gradient(self, actor_loss, critic_loss):
        grads_actor = self.actor_tape.gradient(
            actor_loss, self.actor_model.trainable_variables)

        grads_critic = self.critic_tape.gradient(
            critic_loss, self.critic_model.trainable_variables)

        return grads_actor, grads_critic

    def compute_loss(self, rewards):
        with self.actor_tape:
            with self.critic_tape:

                eps_len = len(rewards)
                expected_rewards = self.get_expected_return(rewards)
                advantage = self.advantage_compute(
                    self.critic_predicts, expected_rewards)

                actions_log = self.actions_log_compute(self.actor_predicts)
                actor_loss = -tf.reduce_sum(actions_log * advantage)

                critic_loss = self.huber_loss(
                    self.critic_predicts, expected_rewards)
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
                self.actor_predicts = []
                self.critic_predicts = []


num_hidden_units = 2048


class Worker():
    def __init__(self, port, subprocess=True):

        if subprocess:
            args = ['unity_game/HarryAndTheStone.exe', '--p', str(port)]
            self.subprocess = subprocess.Popen(args)

        self.env = Unity(port)

        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

        self.harry_a2c = ActorCritic(optimizer, 6, num_hidden_units, [3, 3])

        self.first_thieve_a2c = ActorCritic(
            optimizer, 8, num_hidden_units, [3, 3, 2])
        self.second_thieve_a2c = ActorCritic(
            optimizer, 8, num_hidden_units, [3, 3, 2])
        """
        self.first_thieve_a2c_grab = ActorCritic(
            optimizer, 2, num_hidden_units)
        self.second_thieve_a2c_grab = ActorCritic(
            optimizer, 2, num_hidden_units)
        """

    def apply_harry(self, frame_action, harry_state, fully_state):

        actions, _ = self.harry_a2c.predict(harry_state, fully_state)

        x = int(actions[0]) - 1
        y = int(actions[1]) - 1

        self.env.apply_harry_action(frame_action, x, y)

    def grab_converter(self, grab):
        if grab == 0:
            return 0
        else:
            return 1

    def apply_thieve(self, frame_action, thieve_id, thieve_state, fully_state):

        if thieve_id == 0:
            actions, _ = self.first_thieve_a2c.predict(
                thieve_state, fully_state, False)
        else:
            actions, _ = self.second_thieve_a2c.predict(
                thieve_state, fully_state)

        x = int(actions[0]) - 1
        y = int(actions[1]) - 1
        grab = int(actions[2])
        if thieve_id == 0:
            self.env.apply_first_thieve_action(frame_action, x, y, grab)
        else:
            self.env.apply_second_thieve_action(frame_action, x, y, grab)

    def run_episode(self):
        state = self.env.reset()
        harry_rewards = []
        first_thieve_rewards = []
        second_thieve_rewards = []

        first_thieve_done = False
        second_thieve_done = False

        while(True):

            frame_action = {}

            full_state, harry_state, fir_thieve_state, sec_thieve_state = state

            self.apply_harry(frame_action, harry_state, full_state)

            if not first_thieve_done:
                self.apply_thieve(
                    frame_action, 0, fir_thieve_state, full_state)

            if not second_thieve_done:
                self.apply_thieve(
                    frame_action, 1, sec_thieve_state, full_state)

            state, rewards, done = self.env.action(frame_action)

            if not first_thieve_done:
                first_thieve_rewards.append(rewards[1])

            if not second_thieve_done:
                second_thieve_rewards.append(rewards[2])

            harry_rewards.append(rewards[0])

            if done[0]:
                return harry_rewards, first_thieve_rewards, second_thieve_rewards

            if done[1]:
                first_thieve_done = True

            if done[2]:
                second_thieve_done = True

    def train(self, max_ep):
        for i in range(max_ep):
            harry_reward, first_thieve_rewards, second_thieve_rewards = self.run_episode()

            self.harry_a2c.apply_loss(harry_reward)
            self.first_thieve_a2c.apply_loss(first_thieve_rewards)
            self.second_thieve_a2c.apply_loss(second_thieve_rewards)

            global eps_count
            eps_count += 1
            print("Done", eps_count)

        self.close()

    def close(self):
        if subprocess:
            self.subprocess.terminate()


#tf.random.set_seed(seed)
#np.random.seed(seed)

print("Start learning")

max_episodes = 1000
max_steps_per_episode = 1000
running_reward = 0


def main():

    worker = Worker(7979, subprocess=False)
    worker.train(100000)

    import time
    t = time.time()

    port = random.randint(7000, 60000)

    #100

    worker_count = 1
    worker_eps_count = 1000

    workers = [Worker(port + i) for i in range(worker_count)]
    threads = [threading.Thread(target=workers[i].train, args=(
        worker_eps_count,)) for i in range(worker_count)]

    for i in range(worker_count):
        threads[i].start()

    for i in range(worker_count):
        threads[i].join()

    print("done", time.time() - t)


if __name__ == "__main__":
    main()
