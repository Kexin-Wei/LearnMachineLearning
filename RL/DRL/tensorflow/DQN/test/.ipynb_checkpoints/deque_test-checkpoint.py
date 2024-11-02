from collections import deque
import numpy as np
l = deque(maxlen = 3)
l.append(np.random.rand(3,2))
l.append(np.random.rand(3,2))
l.append(np.random.rand(3,2))
print(l)
l.append([0,0,0])
print(len(l))