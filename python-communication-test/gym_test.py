import socket
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def eof_check (message):
    eof_format = "<EOF>"
    message_len = len(message)
    if message_len > 5:
        for i in range(5):
            if eof_format[i] != message[message_len + i - 5]:
                return False
        return True
    return False

def get_message ():
    message = ""
    while (True):
        data = clientsocket.recv(1024)
        message += (data.decode('UTF-8'))
        if eof_check(message):
            message = message[:-5]
            break
        
    return message

def send_message (client , message):
    message += "<EOF>"
    client.send(bytearray(message , 'UTF-8'))


SERVER = 'localhost'
PORT = 7979

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((SERVER , PORT))

server.listen()

(clientsocket, address) = server.accept()



actions = ["left" , "right" , "up" , "down"]

def action_map (action_id):
    return actions[action_id]
def state_map (state):
    arr = []
    for i in state.values():
        arr.append(i)
    return arr

seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

num_inputs = 4
num_actions = 4
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

test = {}

while (True):

    callback_message = {}
    callback_message["isEnd"] = True

    json_respond = json.dumps(callback_message) 
    send_message(clientsocket , json_respond)
    
    message = get_message()
    json_message =  json.loads(message)

    state = state_map(json_message["state"])
    
    episode_reward = 0

    with tf.GradientTape() as tape:
        while(True):
            
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            
            callback_message = {}
            callback_message["harryCommand"] = action_map(action)
            json_respond = json.dumps(callback_message) 
            send_message(clientsocket , json_respond)
            
            if test.get(action , None) == None:
                test[action] = 1
            else:
                 test[action] += 1

            message = get_message()
            json_message =  json.loads(message)

            reward = int(json_message["lastReward"])
            state = state_map(json_message["state"])
            done =  json_message["isEnd"]
            

            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break
            # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    #if running_reward > 195:  # Condition to consider the task solved
    #    print("Solved at episode {}!".format(episode_count))
    #    break