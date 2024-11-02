import numpy as np
import matplotlib.pyplot as plt

n=10
epsilon=0.3
p=np.random.dirichlet(np.ones(n))*(1-epsilon)
p+=epsilon/n
plt.bar(np.arange(n)+1,p)
plt.show()