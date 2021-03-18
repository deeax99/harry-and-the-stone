import socket
import json
import random
import qlearning as ql

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

def randomAction ():
    dic = {}
    if random.random() > .5:
        dic["harryCommand"] = "left"
    else : 
        dic["harryCommand"] = "right"

    return dic

prevState = None
prevAction = None
prevReward = 0

actions = ["left" , "right" , "up" , "down" , "stop"]

q = ql.QLearn(actions) 

while (True):

    message = get_message()
    json_message =  json.loads(message)
    prevReward = int(json_message["lastReward"])

    del json_message['lastReward']
    state = json.dumps(json_message)

    if prevState != None :
        q.learn(prevState , prevAction , prevReward , state)
    
    prevAction = q.chooseAction(state)
    prevState = state
    
    callback_message = {}
    callback_message["harryCommand"] = prevAction

    json_action = json.dumps(callback_message) 
    send_message(clientsocket , json_action)