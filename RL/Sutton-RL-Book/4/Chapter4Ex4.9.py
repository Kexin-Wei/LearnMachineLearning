"""
created by Vac, 2020.8.11
solve Ex4.9 from RL book
the gambler problem:
    have 1-99 bucks, win with 100 or over bucks
    coin head win beat*2, coin tail lose beat
    reward get when end with 100 or over bucks
Value Iteration for policy optimalization

Algorithm:

1. Initialization
    V(s) in R and pi(s) in A(s) arbitrarily for all s in S+, arbitrarily except that V (terminal) = 0

2. Value Iteration
    Loop:
        delta<-0
        for each s in S:
            v<-V(s)
            V(s)<-max_a(sum(p(s',r|s,a)*(r + gamma* V (s')))
            delta<-max( delta , |v-V(s)|)
    until  delta < theta (a small positive number determining the accuracy of estimation)

3. Output a deterministic policy, such that
    pi(s) = argmax(sum(p(s',r|s,a)*(r + gamma* V (s')))

"""
import matplotlib.pyplot as plt
import numpy as np
import os


class Gambler:

    def __init__(self, theta=0.03, gamma=0.9, ph=0.4):
        # state: 0,1,2,3,4....98
        # money: 1,2,3,4,5....99
        self.V = [0 for i in range(99)]
        self.policy = [0 for i in range(99)]
        self.Terminal = [-1, 99]  # state end =99, money >=100
        # or state end =-1, money<0
        self.delta = 0
        self.theta = theta
        self.gamma = gamma
        self.ph = ph

    def isTerminal(self, s):
        for T in self.Terminal:
            if s == T:
                return True
        return False

    def StateMove(self, s, a, win=True):
        if win:
            money = s + 1 + a + 1  # coin head get same as beat
        else:
            money = s - a  # coin tail lose beat

        # check if state terminal
        if money >= 100: return 99  # end state 99, money >=100
        if money <= 0: return -1  # end up no money

        return money - 1  # state:0,1,2.... money:1,2,3...

    def getreward(self, s_next):
        # only if the money > 100, state=99
        if s_next == 99: return 100
        return 0

    def max_a(self, s):
        choice = np.zeros(s + 1)
        for a in range(s + 1):
            # action: 0,1,2,3...
            # beat:   1,2,3,4...

            # every action will have 2 results:

            # 1. win
            s_next_win = self.StateMove(s, a, win=True)  # s=3,a=2, have money 4 win 3, money will be 7, state = 6
            reward = self.getreward(s_next_win)
            if self.isTerminal(s_next_win):  # next is terminal, 0 -> V(s')
                choice[a] += self.ph * (reward + self.gamma * 0)
            else:
                choice[a] += self.ph * (reward + self.gamma * self.V[s_next_win])

            # 2. lose
            s_next_lose = self.StateMove(s, a, win=False)  # s=3,a=2, have money 4 lose 3, money 1, state 0
            reward = self.getreward(s_next_lose)
            if self.isTerminal(s_next_lose):  # next is terminal, 0 -> V(s')
                choice[a] += (1 - self.ph) * (reward + self.gamma * 0)
            else:
                choice[a] += (1 - self.ph) * (reward + self.gamma * self.V[s_next_lose])
        return max(choice)

    def argmax(self, choice, method):
        # method: random(default): choose one of the max
        #         min:             choose the action min one
        max_2 = round(max(choice), 2)
        a_range = len(choice)
        candidate = []
        for a in range(a_range):
            if round(choice[a], 2) == max_2: candidate.append(a)
        maybe = np.random.randint(len(candidate))
        if method == "min": return candidate[0]
        return candidate[0]

    def argmax_a(self, s, method):
        # method description in function argmax()
        choice = np.zeros(s + 1)
        for a in range(s + 1):
            # action: 0,1,2,3...
            # beat:   1,2,3,4...

            # every action will have 2 results:

            # 1. win
            s_next_win = self.StateMove(s, a, win=True)  # s=3,a=2, have money 4 win 3, money will be 7, state = 6
            reward = self.getreward(s_next_win)
            if self.isTerminal(s_next_win):  # next is terminal, 0 -> V(s')
                choice[a] += self.ph * (reward + self.gamma * 0)
            else:
                choice[a] += self.ph * (reward + self.gamma * self.V[s_next_win])

            # 2. lose
            s_next_lose = self.StateMove(s, a, win=False)  # s=3,a=2, have money 4 lose 3, money 1, state 0
            reward = self.getreward(s_next_lose)
            if self.isTerminal(s_next_lose):  # next is terminal, 0 -> V(s')
                choice[a] += (1 - self.ph) * (reward + self.gamma * 0)
            else:
                choice[a] += (1 - self.ph) * (reward + self.gamma * self.V[s_next_lose])
        return self.argmax(choice, method)

    def figPolicy(self, figdir='fig4.9'):
        fig = plt.figure()
        plt.bar(range(99) + np.ones(99), self.policy)
        title = 'theta=' + str(self.theta) + ', gamma=' + str(self.gamma) + ', ph=' + str(self.ph)
        plt.title(title)
        # figname=figdir+'/ph'+str(self.ph)+'_gamma'+str(self.gamma)+'_theta'+str(self.theta)+'.png'
        # plt.savefig(figname)
        plt.savefig(figdir)
        plt.show(block=False)
        plt.pause(1)

    def ValueIteration(self, method="random"):
        iteration = 0
        while (self.delta > self.theta or iteration == 0):
            print("ValueIteration: ", iteration)
            iteration += 1
            self.delta = 0
            for s in range(99):
                v = self.V[s]
                self.V[s] = self.max_a(s)
                self.delta = max(self.delta, abs(v - self.V[s]))
        for s in range(99):
            self.policy[s] = self.argmax_a(s, method)
        print('ValueIteration: done. -------------------------------------')


if __name__ == "__main__":
    """
    for i in range(10):
        print("Probability:",i,"*****************************************")
        g=1.0
        t=0.03
        fdir="fig4.9/Probability"+str(i)+'.png'
        gambler=Gambler(theta=t,gamma=g)
        gambler.ValueIteration()
        gambler.figPolicy(figdir=fdir)
    """
    gambler = Gambler(gamma=1.0, ph=0.4)
    gambler.ValueIteration(method="min")
    gambler.figPolicy(figdir="fig4.9/ph0.4_gamma1.png")

    gambler = Gambler(gamma=1.0, ph=0.25)
    gambler.ValueIteration(method="min")
    gambler.figPolicy(figdir="fig4.9/ph0.25_gamma1.png")

    gambler = Gambler(gamma=1.0, ph=0.55)
    gambler.ValueIteration(method="min")
    gambler.figPolicy(figdir="fig4.9/ph0.55_gamma1.png")
