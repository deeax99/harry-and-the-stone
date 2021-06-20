import random
from unity import Unity
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import os 
import subprocess
import math

actions = ["left" , "right" , "up" , "down" ]


episode_reward = 0
trajectorys = []


app_name = 'Environment/HarryAndTheStone.exe'
args = [app_name ]
subprocess.Popen(args, stdout=subprocess.DEVNULL)

env =Unity()
all_reward = []
current_reward=0
done=False
ep=0
all_ep_r = []


class QLearn:
    total_state = 0 
    unique_state = 0
    
    def __init__(self,actions, epsilon=.1, alpha=0.6, gamma=0.7):
        self.q = {}
        self.qvis = {}
        self.epsilon = epsilon  
        self.alpha = alpha      
        self.gamma = gamma
        self.actions=actions

    def retrieve_Q(self, state, action):
        
        return self.q.get((state, action), -3000)
        

    def learn_Q(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def choose_action(self, state):
        self.total_state += 1

        if self.qvis.get(state , None) == None:
            self.unique_state += 1
            self.qvis[state] = 1

        else :
            self.qvis[state] += 1
        rand = random.random()

        if  rand < self.epsilon or (self.total_state / self.unique_state) > self.qvis[state]:
            action = random.randint(0, 3)


        else:
            q = [self.retrieve_Q(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)


            else:
                i = q.index(maxQ)

            action =i


        return action

    def learn_2 (self , trajectorys  , episode_reward):
        for trajectory in trajectorys:
            state = (tuple(trajectory["state"]) , trajectory["action"])
            if self.q.get(state , None) != None:
                self.q[state] =  max(self.q[state] , episode_reward)
            else :
                self.q[state] = episode_reward

            episode_reward -= trajectory["reward"]
            

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.retrieve_Q(state2, a) for a in self.actions])
        self.learn_Q(state1, action1, reward, reward + self.gamma*maxqnew)

q_model = QLearn(actions) 


while (True):
    episode_reward=0
    state = env.reset()
    done=False
    
    while(True):
        state = hash(tuple(state))
        action = q_model.choose_action(state)
        state,current_reward , done =env.action(action)
        
        trajectory = {}

        trajectory["state"] = state
        trajectory["action"] = action
        trajectory["reward"] = current_reward

        trajectorys.append(trajectory)

        episode_reward += current_reward
        
        if done :
            q_model.learn_2(trajectorys , episode_reward)
            all_reward.append(episode_reward)
            state = env.reset()
            
            trajectorys[-1]["reward"] = current_reward
            ep+=1
            all_ep_r.append(episode_reward)
            
            print("total reward = {} episd = {}".format(episode_reward,ep))
            episode_reward = 0
            if ep %10000 ==0:
                plt.plot(np.arange(ep),all_ep_r , color='green')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title('Reward Per Episode Using Q-Learning  ', y=1.1)
                plt.show()

