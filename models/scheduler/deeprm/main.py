"""Entrance."""

import state

if __name__ == '__main__':
    environment, scheduler = env.load()
    print(environment, scheduler)
    while not environment.terminated():
        environment.plot()
        actions = scheduler.schedule()
        print(actions)
    print('END')
