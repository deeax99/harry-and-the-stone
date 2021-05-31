import collections
import numpy as np
import tensorflow as tf
from unity import Unity
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
import os
import math

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


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
        advantage = advantage_compute(values, rewards, epsiodes)
        actions_log = actions_log_compute(action_probs)

        actor_lose = 0
        for i in range(epsiodes):
            actor_lose -= actions_log[i] * advantage[i]

        critic_lose = huber_loss(values, rewards)
        return actor_lose, critic_lose
