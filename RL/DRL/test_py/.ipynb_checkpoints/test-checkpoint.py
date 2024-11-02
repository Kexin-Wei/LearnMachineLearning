# remove a directory
import os 
import shutil

try:
    os.mkdir('test')
except:
    pass
with open('test/test.txt','w') as f:
    f.write("test for the delete")

input("ready?")

shutil.rmtree('test')




# for enumerate
"""
from collections import deque
import numpy as np
import random

d = deque()
for i in range(3):
    d.append([ j for j in range(4)])
    
for index,(l1,l2,l3,l4) in enumerate(d):
    print(index,l1,l2,l3,l4)
"""

# sample deque
"""
from collections import deque
import numpy as np
import random

d = deque()
for i in range(10):
    d.append([ i for j in range(10)])
s = random.sample(d,3)
print(s)
"""

# plot max min ave for eposide
""" 
import matplotlib.pyplot as plt
import numpy as np
big_l = {
    'max':[],
    'min':[],
    'ave':[]
}
for i in range(3):
    small_l = []
    for j in range(10):
        small_l.append(np.random.randint(10))
    print(sum(small_l)/len(small_l),max(small_l),min(small_l))
    big_l['max'].append(max(small_l))
    big_l['min'].append(min(small_l))
    big_l['ave'].append(sum(small_l)/len(small_l))
plt.plot(big_l['max'],label='max')
plt.plot(big_l['min'],label='min')
plt.plot(big_l['ave'],label='ave')
plt.legend()
plt.savefig('test.png')
"""

# write txt
''' 
file = open('text.txt','w')

for i in range(10):
    file.write("Epoch {} - rewards {} and step {}\n".format(i,i,i))

file.close()
'''

#numpy dot
''' 
import numpy as np
inputs = np.random.rand(5)
w = np.random.rand(4,5)

o = np.dot(inputs,w[1])
print(o)

o2 = np.dot(inputs,np.transpose(w))
print(o2[1])
'''


print("DONE")