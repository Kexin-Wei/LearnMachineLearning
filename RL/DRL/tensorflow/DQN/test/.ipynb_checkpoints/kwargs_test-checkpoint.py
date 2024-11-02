from collections import deque
import random

class Replay:
    def __init__(self, MEMORY_SIZE =5_000, BATCH_SIZE = 2):
        self.BATCH_SIZE = BATCH_SIZE
        self.memory = deque([],maxlen = MEMORY_SIZE)
    def __call__(self, a_set_memory):
        # a_set_memory = sars(a) : [ob, act, reward, ob_next, done]
        self.memory.append(a_set_memory)
    def sample(self,*args):
        if args:
            b_size = args[0]
        else:
            b_size = self.BATCH_SIZE
        return random.sample(self.memory,b_size)
        
a = Replay()
for i in range(11):
    a([i for j in range(4)])
print(a.memory)
print(a.sample())
print(a.sample(3))