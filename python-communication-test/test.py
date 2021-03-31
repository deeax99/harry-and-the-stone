import socket
import json
import random
import qlearning as ql
from time import sleep

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


def send_message (client , message):
    message += "<EOF>"
    client.send(bytearray(message , 'UTF-8'))


SERVER = 'localhost'
PORT = 7979

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((SERVER , PORT))

server.listen()

(clientsocket, address) = server.accept()



actions = ["left" , "right" , "up" , "down" , "stop"]

q = ql.QLearn(actions) 
cumulative_reward = 0
trajectorys = []
firstEnter = False
factor = .9/]/]]/

all_reward = []


while (True):

    message = get_message()
    json_message =  json.loads(message)

    current_reward = int(json_message["lastReward"])

    callback_message = {}

    if firstEnter == False:
        trajectorys = []
        cumulative_reward = 0
        #q.epsilon = random.random() * factor
        firstEnter = True
    else :
        trajectorys[-1]["reward"] = current_reward
        cumulative_reward += current_reward
    
    if json_message["isEnd"] == True :
        q.learn_2(trajectorys , cumulative_reward)
        firstEnter = False
        callback_message["isEnd"] = True
        all_reward.append(cumulative_reward)     
    else :
        state = json.dumps(json_message["state"])
        action = q.chooseAction(state)
        
        trajectory = {}
        
        trajectory["state"] = state
        trajectory["action"] = action

        trajectorys.append(trajectory)
        
        callback_message["harryCommand"] = action

    
    json_respond = json.dumps(callback_message) 
    send_message(clientsocket , json_respond)
