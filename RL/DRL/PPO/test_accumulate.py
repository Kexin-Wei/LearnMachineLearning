# %%
from itertools import accumulate
import numpy as np
import operator
#%%
td = [0.5]*10;td.insert(0,1)
print(td)
ad = list(accumulate(td,operator.mul))
print(ad)
# %%
td = [1,2,3,4,5]
print(td[::-1])
print(list(accumulate(td[::-1],lambda x,y:x*.5+y)))
#ad = list(accumulate())
# %%
td = td.remove(-1)
# %%
