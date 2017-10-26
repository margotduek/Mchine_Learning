import numpy as np



def sigmoid(X):
    A = []
    a = []
    for i in range(len(X)):
        #for j in range(len(X[i])):
        c = 1.0 / (1.0 + (np.exp(-X[i])))
        A.append(c)
        #A.append("||")
        #A.append(a)

    print(A)
    return A


sigmoid([.31, .44, .37, .48])
