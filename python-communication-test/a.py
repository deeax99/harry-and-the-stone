from unity import Unity
env = Unity()

limited = 0
while (True):
    print("start state",env.reset())
    while (True):
        action = {}
        Unity.apply_second_thieve_action(action ,1,0,1)
        if (limited % 1000 == 0):
            print("test",env.reset())
            
        state , _ , _ = env.action(action)

            
        limited += 1
        if limited == 1000:
            print("reset")
            limited = 0
            