choice=[0.123124,0.12354,0.03451,0.123145]
pi=[1,1,1,1]


def argmax_a(choice,Pi):
    # find the max+-0.01, when the rest is smaller, put it to 0
    max_expection=round(max(choice),2)
    pi_new=[0.0, 0.0, 0.0, 0.0]
    for a in range(4):
        if round(choice[a],2) == max_expection: pi_new[a]=Pi[a]
        else: pi_new[a]=0
    return pi_new

print(argmax_a(choice,pi))