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
import signal
import sys


def handler(signum, frame):
    print("Bye")
    sys.exit(0)


signal.signal(signal.SIGINT, handler)

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)


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
    def __init__(self, optimizer, num_actions, num_hidden_units):

        self.actor_model = Actor(num_actions, num_hidden_units)
        self.critic_model = Critic(num_hidden_units)

        self.optimizer = optimizer

        self.clear()

    def predict(self, actor_state, fully_state):

        actor_state = tf.expand_dims(actor_state, 0)
        fully_state = tf.expand_dims(fully_state, 0)

        with self.actor_tape:
            with self.critic_tape:

                actor_predict = self.actor_model(actor_state)
                actor_predict_logits = tf.nn.softmax(actor_predict)
                action = self.get_action(actor_predict)[0,0]

                critic_predict = self.critic_model(fully_state)

                self.actor_predicts.append(actor_predict_logits[0,action])
                self.critic_predicts.append(critic_predict[0, 0])

                return action, critic_predict[0, 0]

    def apply_loss(self, rewards):

        actor_loss, critic_loss = self.compute_loss(rewards)

        grads_actor = self.actor_tape.gradient(
            actor_loss, self.actor_model.trainable_variables)

        grads_critic = self.critic_tape.gradient(
            critic_loss, self.critic_model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads_actor, self.actor_model.trainable_variables))

        self.optimizer.apply_gradients(
            zip(grads_critic, self.critic_model.trainable_variables))

        self.clear()

    def compute_loss(self, rewards):
        with self.actor_tape:
            with self.critic_tape:

                eps_len = len(rewards)
                expected_rewards = self.get_expected_rewards(rewards)
                advantage = self.advantage_compute(
                    self.critic_predicts, rewards)

                actions_log = self.actions_log_compute(self.actor_predicts)
                actor_loss = -tf.reduce_sum(actions_log * advantage)

                critic_loss = huber_loss(self.critic_predicts, rewards)
                return actor_loss, critic_loss

    def get_expected_rewards(self, reward):
        exp_reward = []
        reward_sum = sum(reward)

        for i in range(len(reward)):
            exp_reward.append(reward_sum)
            reward_sum -= reward[i]

        return exp_reward

    def advantage_compute(self, values, rewards):
        advantage = []
        framecount = len(rewards)
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


num_hidden_units = 512
env = Unity()

#TODO save
harry_a2c = ActorCritic(optimizer, 4, num_hidden_units)
first_thieve_a2c = ActorCritic(optimizer, 4, num_hidden_units)
second_thieve_a2c = ActorCritic(optimizer, 4, num_hidden_units)

first_thieve_a2c_grab = ActorCritic(optimizer, 2, num_hidden_units)
second_thieve_a2c_grab = ActorCritic(optimizer, 2, num_hidden_units)


def apply_harry(frame_action, harry_state, fully_state):

    action, _ = harry_a2c.predict(harry_state, fully_state)
    x, y = env.get_movement_action(action)
    env.apply_harry_action(frame_action, x, y)


def grab_converter(grab):
    if grab == 0:
        return 0
    else:
        return 1


def apply_thieve(frame_action, thieve_id, thieve_state, fully_state):

    if thieve_id == 0:
        action, _ = first_thieve_a2c.predict(thieve_state, fully_state)
        grab, _ = first_thieve_a2c_grab.predict(thieve_state, fully_state)
    else:
        action, _ = second_thieve_a2c.predict(thieve_state, fully_state)
        grab, _ = second_thieve_a2c_grab.predict(thieve_state, fully_state)

    x, y = env.get_movement_action(action)

    grab = grab_converter(grab)
    if thieve_id == 0:
        env.apply_first_thieve_action(frame_action, x, y, grab)
    else:
        env.apply_second_thieve_action(frame_action, x, y, grab)


def run_episode():
    state = env.reset()
    harry_rewards = []
    thieve_rewards = []

    first_thieve_a2c_done = False
    second_thieve_done = False

    while(True):

        frame_action = {}

        full_state, harry_state, fir_thieve_state, sec_thieve_state = state

        apply_harry(frame_action, harry_state, full_state)

        if not first_thieve_a2c_done:
            apply_thieve(frame_action, 0, fir_thieve_state, full_state)

        if not second_thieve_done:
            apply_thieve(frame_action, 1, sec_thieve_state, full_state)

        state, rewards, done = env.action(frame_action)

        harry_rewards.append(rewards[0])
        thieve_rewards.append(rewards[1])

        if done[0]:
            return harry_rewards, thieve_rewards

        if done[1]:
            first_thieve_a2c_done = True

        if done[2]:
            second_thieve_done = True


def train(max_ep):
    for i in range(max_ep):
        harry_reward, thieve_reward = run_episode()

        harry_a2c.apply_loss(harry_reward)

        first_thieve_a2c.apply_loss(thieve_reward)
        second_thieve_a2c.apply_loss(thieve_reward)
        first_thieve_a2c_grab.apply_loss(thieve_reward)
        second_thieve_a2c_grab.apply_loss(thieve_reward)


seed = 156
tf.random.set_seed(seed)
np.random.seed(seed)

print("Start learning")

max_episodes = 1000
max_steps_per_episode = 1000
running_reward = 0


def main():
    train(10000)


if __name__ == "__main__":
    main()
