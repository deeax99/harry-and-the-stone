from unity import Unity
env = Unity()

limited = 0
while (True):
    print("start state", env.reset())
    while (True):
        action = {}

        Unity.apply_second_thieve_action(action, 1, 0, 1)
        if (limited % 1000 == 0):
            print("test")
        limited += 1
        state, rewards, dones = env.action(action)
        if dones[0]:
            print(dones , rewards)
            env.reset()
            limited = 0
