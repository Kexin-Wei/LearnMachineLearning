# %%
# from tqdm import tqdm
# import time

# loss = 0.2343
# kl = 0.948394
# for i in tqdm(range(100)):
#     time.sleep(1)
#     tqdm.write(f'\tkl: {kl:.2f} \tloss {loss:.2f}')
    # bar on bottom, words flash
# %%

import time
from tqdm import tqdm

pbar = tqdm(range(20))
loss = 100
kl   = 100
for i in pbar:
    time.sleep(1)
    loss+=1;kl-=1
    if i%2==0: pbar.write(f"Now ep:{i}")
    pbar.set_postfix_str(s=f"kl:{kl} loss:{loss}")

