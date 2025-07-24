from client.rlenv import KinovaEnv

if __name__ == "__main__":
    env = KinovaEnv()

    while True:
        cmd = input("Enter command:")
        cmds = cmd.split(' ')
        if cmds[0] == 'a':
            # read in action deltas
            action = list(map(float, cmds[1:]))
            obs, rew, succes = env.step(action)
            print("stepped")
            print(f"Observation: {obs}")
            print(f"Reward: {rew}")
            print(f"Success: {succes}")
        if cmds[0] == 'r':
            env.reset()
            print("Reset")
        if cmds[0] == 'b':
            break
