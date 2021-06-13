from tcp import TCPUitility
from tcp import TCPConnection
import numpy as np

class Unity:
    def __init__(self , client = None):
        if client:
            self.client = client
        else:
            self.tcp_connection = TCPConnection(7979)
            self.client = self.tcp_connection.get_client() 

    harry_actions = ["left", "right", "up", "down"]

    def harry_action_map(self, action_id):
        return Unity.harry_actions[action_id]

    def state_map(self, state):
        arr = []
        for i in state.values():
            arr.append(i)
        return arr

    def action(self, harry_action):

        ml_message = {}
        ml_message["harryCommand"] = self.harry_action_map( harry_action)

        TCPUitility.send_message(self.client, ml_message)

        unity_message = TCPUitility.get_message(self.client)

        state = self.state_map(unity_message["state"])
        reward = int(unity_message["lastReward"])
        done = unity_message["isEnd"]
        state = np.asarray(state)

        return state, reward, done

    def reset(self):
        ml_message = {}
        ml_message["isEnd"] = True

        TCPUitility.send_message(self.client, ml_message)

        unity_message = TCPUitility.get_message(self.client)

        state = self.state_map(unity_message["state"])
        state = np.asarray(state)
        return state

    def destroy(self):
        self.tcp_connection.destroy()
