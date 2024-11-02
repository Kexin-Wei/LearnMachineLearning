# test
Terminal=[[0,3],[3,0]]

def isTerminal(i,j,Terminal):
        for T in Terminal:
            if i == T[0] and j ==T[1]:
                return True
        
        return False
print(isTerminal(3,0,Terminal))           

import matplotlib.pyplot as plt
import numpy as np

V = [[0 for i in range(4)]for i in range(4)]
pi_s = [0.25, 0.25, 0.25, 0.25]
Pi = [[pi_s for i in range(4)] for i in range(4)]
def figPlot():    
    fig=plt.figure(figsize=(4,4))
    plt.xlim([0,4])
    plt.ylim([0,4])
    ax=fig.gca()
    ax.set_xticks(np.arange(0,5,1))
    ax.set_yticks(np.arange(0,5,1))
    plt.grid()  #set grid

    ax = fig.add_subplot(111)
    plt.gca().add_patch(plt.Rectangle((0,3),1,1,color="black"))
    plt.gca().add_patch(plt.Rectangle((3,0),1,1,color="black"))# set rectangle

    for i in range(4):
        for j in range(4):
            for k in range(4):
                if Pi[i][j][k]!=0:
                    v=V[i][j]
                    if k==0: plt.annotate(text=str(v),xy=(i,j+0.5),xytext=(i+0.4,j+0.4),arrowprops={"arrowstyle":"->"}) #left
                    if k==1: plt.annotate(text=str(v),xy=(i+0.5,j+1),xytext=(i+0.4,j+0.4),arrowprops={"arrowstyle":"->"}) #up
                    if k==2: plt.annotate(text=str(v),xy=(i+1,j+0.5),xytext=(i+0.4,j+0.4),arrowprops={"arrowstyle":"->"}) #right
                    if k==3: plt.annotate(text=str(v),xy=(i+0.5,j),xytext=(i+0.4,j+0.4),arrowprops={"arrowstyle":"->"}) #down
    plt.show()

figPlot()