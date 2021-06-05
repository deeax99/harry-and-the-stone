from tcp import TCPUitility
from tcp import TCPConnection


class Unity:

    @classmethod
    def get_movement_action(cls,action):
        if action == 0:
            return -1,0
        elif action == 1:
            return 1,0
        elif action == 2:
            return 0,1
        else:
            return 0,-1
            
    @classmethod
    def apply_harry_action (cls,action_message , harry_x, harry_y):
        action_message["harryX"] = harry_x
        action_message["harryY"] = harry_y

    @classmethod
    def apply_first_thieve_action (cls,action_message ,thieve_x, thieve_y , thieve_grab):
        action_message["firstThieveX"] = thieve_x
        action_message["firstThieveY"] = thieve_y
        action_message["firstThieveGrab"] = thieve_grab

    @classmethod
    def apply_second_thieve_action (cls,action_message ,thieve_x, thieve_y , thieve_grab):
        action_message["secondThieveX"] = thieve_x
        action_message["secondThieveY"] = thieve_y
        action_message["secondThieveGrab"] = thieve_grab
    

    def __init__(self , client = None):
        if client:
            self.client = client
        else:
            self.tcp_connection = TCPConnection(7979)
            self.client = self.tcp_connection.get_client() 

    def get_state(self , unity_message):
        return unity_message["fullState"] ,unity_message["harryState"] , unity_message["firstThieve"] , unity_message["secondThieve"]

    def action(self, action):

        TCPUitility.send_message(self.client, action)

        unity_message = TCPUitility.get_message(self.client)

        state = self.get_state(unity_message)
        
        reward = ((int)(unity_message["harryReward"]) , (int)(unity_message["thievesReward"]))
        done = (unity_message["done"] , unity_message["firstThieveEnd"] , unity_message["secondThieveEnd"])

        return state, reward, done

    def reset(self):
        ml_message = {}
        ml_message["done"] = True

        TCPUitility.send_message(self.client, ml_message)

        unity_message = TCPUitility.get_message(self.client)

        state = self.get_state(unity_message)

        return state

    def destroy(self):
        self.tcp_connection.destroy()
