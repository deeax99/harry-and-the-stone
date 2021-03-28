import random


class QLearn:
    total_state = 0 
    unique_state = 0
    
    def __init__(self, actions, epsilon=.1, alpha=0.6, gamma=0.7):
        self.q = {}
        self.qvis = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma
        self.actions = actions
    # 
    # table access 
    def getQ(self, state, action):
        return self.q.get((state, action), -3000)
        # return self.q.get((state, action), 1.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:        
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state):
        self.total_state += 1
        if self.qvis.get(state , None) == None:
            self.unique_state += 1
            self.qvis[state] = 1
        else :
            self.qvis[state] += 1
        rand = random.random()
        if  rand < self.epsilon or (self.total_state / self.unique_state) > self.qvis[state]:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            # In case there're several state-action max values 
            # we select a random one among them
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action

    def learn_2 (self , trajectorys  , cumulative_reward):
        for trajectory in trajectorys:
            state = (trajectory["state"] , trajectory["action"])
            if self.q.get(state , None) != None:
                self.q[state] =  max(self.q[state] , cumulative_reward)
            else : 
                self.q[state] = cumulative_reward

            cumulative_reward -= trajectory["reward"]
            

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

import math
def ff(f,n):
    fs = "{:f}".format(f)
    if len(fs) < n:
        return ("{:"+n+"s}").format(fs)
    else:
        return fs[:n]
