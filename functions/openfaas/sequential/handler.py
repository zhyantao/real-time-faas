import time
import json
import random

def handle(args):
    start_time = time.time()
    time.sleep(1) # s
    
    if True:
        func2()
    else:
        func3()
    
    end_time = time.time()
    return json.dumps({'start_time': start_time, 'end_time': end_time})

def func2():
    time.sleep(1)
    func4()
    
    
def func3():
    time.sleep(1)
    func4()
    
def func4():
    time.sleep(1)