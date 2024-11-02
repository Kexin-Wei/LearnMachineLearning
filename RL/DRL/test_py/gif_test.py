import imageio
import os
DIR = 'gym_graph'
images = []
for f in os.listdir(DIR):
    if f.startswith("2_"):
        images.append(imageio.imread(os.path.join(DIR,f)))
imageio.mimsave('2.gif',images)