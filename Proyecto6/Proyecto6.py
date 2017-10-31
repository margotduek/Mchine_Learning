import random
import numpy as np
import csv
import matplotlib.pyplot as plt
from pylab import plot, ylim


# def lee_numeros(filename):
#   x = []
#   y = []
#   with open(filename, 'r') as file:
#     reader = csv.reader(file, delimiter=' ')
#     for row in reader:
#       filtered = list(filter(lambda x: x != "", row))
#       if filtered:
#         x.append(list((map(lambda x : float(x), filtered[:-1]))))
#         y.append(int(filtered[-1]))
#
#   return (np.mat(x).T, np.mat(y))



def lee_numeros(archivo):
    X = np.zeros((400, 5000))
    y = np.zeros((1, 5000))

    with open(archivo, 'r') as a:
        column = 0
        for line in a:
            cols = list(filter(lambda x: x != "", line.split(" ")))
            row = 0
            for i in range(len(cols) - 1):
                X[row][column] = float(cols[i])
                print(X[row][column])
                row += 1
                y[0][column] = float(cols[400])
            column += 1
    # print(y)
    print(X)

    temp = []
    temp = np.zeros((10, 5000))
    print(len(y))
    print("y0", len(y[0]))
    for i in range(len(y[0])):
        yi = int(y[0][i]) - 1
        temp[yi][i] = 1
        #print(y[0][i])
    # print(temp.transpose())

    return X, temp.transpose()



def sigmoidal(x):
    return 1/(1+np.exp(-x))

def entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y):
    alpha = 2
    ## Forward Propagation ##
    W1 = randInicializacionPesos(25, 400)
    b1 = np.zeros((25, 1))
    # Z1 = W1X + B1
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoidal(Z1)

    b2 = np.zeros((10, 1))
    W2 = randInicializacionPesos(10, 25)
    # Z2 = A1W2 + B2
    Z2 = np.dot(W2, A1) + b2
    #ygorrito = A2 y gorrito es la salida final, como en este caso a2 es la final por eso es ygorrito
    ygorrito = sigmoidal(Z2)

    ## Backward Propagation ##
    dz2 = ygorrito - y
    dw2 = np.dot(dz2, A1.transpose()) * (1/num_labels)
    db2 = np.sum(dz2, axis=0) * (1/num_labels)
    W2 = W2 - alpha * dw2
    b2 = b2 - alpha * db2

    dz1 = np.multiply(np.dot(dw2.transpose(), dz2), sigmoidalGradiente(Z1))
    dw1 = np.dot(dz1, X.transpose())
    db1 = np.sum(dz1, axis=0) * (1/num_labels)
    W1 = W1 - alpha * dw1
    b1 = b1 - alpha * db1

    return W1, b1, W2, b2


def sigmoidalGradiente(z):
    return np.multiply((1/(1+np.exp(-z))),(1-(1/(1+np.exp(-z)))))


def randInicializacionPesos(L_in, L_out):
    W = np.zeros((L_in, L_out))

    for i in range(L_in):
        for j in range(L_out):
            W[i][j] = random.uniform(-0.12, 0.12)

    return W

def prediceRNYaEntrenada(X,W1,b1,W2,b2):
    lasty = []
    # Z1 = W1X + B1
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoidal(Z1)
    # Z2 = A1W2 + B2
    Z2 = np.dot(W2, A1) + b2
    #ygorrito = A2 y gorrito es la salida final, como en este caso a2 es la final por eso es ygorrito
    ygorrito = sigmoidal(Z2)
    for i in range(len(ygorrito)):
        for j in range(len(ygorrito[i])):
            maxi = max(ygorrito[i])
            if(ygorrito[i][j] == maxi):
                lasty.append(j)

    return lasty

X, y = lee_numeros("digitos.txt")
W1, b1, W2, b2 = entrenaRN(400, 25, 10, X, y)
ygorrito = prediceRNYaEntrenada(X,W1,b1,W2,b2)
# np.set_printoptions(threshold=np.nan)

print(y)
print(ygorrito)
print(y.shape)
print(ygorrito.shape)
