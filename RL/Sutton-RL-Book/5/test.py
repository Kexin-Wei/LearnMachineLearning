import numpy as np

def main():
    Value=np.random.randint(2,size=(4,4,4))/2

    a_Value=np.zeros((4,4,4))
    for i in range(4):
        for j in range(4):
            non_zero_index=np.nonzero(Value[i,j])
            off_set=[ 1 if k in non_zero_index[0] else 0 for k in range(4)]
            p=np.random.rand(4)+np.array(off_set)
            p=p/sum(p)
            a_Value[i,j]=p
    check_sum(a_Value)


def check_sum(Policy): 
    for i in range(4):
        for j in range(4):
            if sum(Policy[i,j])==1:
                Value=True
            else:
                Value=False
                print(sum(Policy[i,j]))
            print('Position (',i,",",j,") is", Value)

if __name__=="__main__":
    main()


