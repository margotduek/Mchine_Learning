import random
import numpy as np
import csv
import matplotlib.pyplot as plt
from pylab import plot, ylim


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
                row += 1
                y[0][column] = float(cols[400])
            column += 1

    temp = []
    temp = np.zeros((10, 5000))
    for i in range(len(y[0])):
        yi = int(y[0][i]) - 1
        temp[yi][i] = 1
        #print(y[0][i])
    # print(temp.transpose())

    return X, temp.transpose()


def sigmoidal(x):
    return 1/(1+np.exp(-x))

def entrenaRN(input_layer_size, total_layers, hidden_layer_size, num_labels, X, y):
    dict = {}
    alpha = 2
    dict['A0'] = X

    for ii in range(total_layers):
        # Inicializar W y b
        dict['W' + str(ii+1)] = randInicializacionPesos(total_layers[ii], total_layers[ii])
        dict['b' + str(ii+1)] = np.mat(np.zeros(total_layers[ii+1]))

        for i in range(1000):
            ## Forward Propagation ##
            for i in range(layers):
                # Hacer el cálculo de Z y de A
                dict['Z' + str(i+1)] = dict['A' + str(i)].dot(dict['W' + str(i+1)].T)+ dict['b' + str(i+1)]
                dict['A' + str(i+1)] = sigmoidal(dict['Z' + str(i+1)])

        ## Backward Propagation ##

        for i in range(total_layers,0,-1):
            # Tenemos que revisar si es la úlima capa en la que estamos haciendo las operaciones
            # si si es diferente entonces la operacion varía un poco
            if(i + 1 == total_layers):
                dict['dZ' + str(i)] = (dict['A' + str(i)] - dict['y'])
            else:
                dict['dZ' + str(i)] = np.multiply(dict['dZ' + str(i+1)].dot(dict['W' + str(i+1)]),sigmoidalGradiente(dict['Z' + str(i)]))

            # Calculamos dw y db
            dict['dW' + str(i)] = (1/m) * dict['dZ' + str(i)].T.dot(dict['A' + str(i-1)])
            dict['db' + str(i)] = (1/m) * np.sum(dict['dZ' + str(i)], axis=0)

            dict['W' + str(i)] = dict['W' + str(i)] - alpha * dict['dW' + str(i)]
            dict['b' + str(i)] = dict['b' + str(i)] - alpha * dict['db' + str(i)]

    return_value = []
    for i in range(total_layers):
        try:
            return_value.append(dict['W' + str(i)])
            return_value.append(dict['b' + str(i)])
        except:
            return dict['W1'], dict['b1'], dict['W2'], dict['b2']



    return return_value



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
    maxi = 0
    # Z1 = W1X + B1
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoidal(Z1)
    # Z2 = A1W2 + B2
    Z2 = np.dot(W2, A1) + b2
    #ygorrito = A2 y gorrito es la salida final, como en este caso a2 es la final por eso es ygorrito
    ygorrito = sigmoidal(Z2)
    #print("gorrito funcion ", ygorrito)
    for i in range(len(ygorrito)):
        for j in range(len(ygorrito[i])):
            maxi = max(ygorrito[i])
            if(ygorrito[i][j] == maxi):
                lasty.append(j)


    return lasty

X, y = lee_numeros("digitos.txt")
W1, b1, W2, b2 = entrenaRN(400, 2, 25, 10, X, y)
ygorrito = prediceRNYaEntrenada(X,W1,b1,W2,b2)
# np.set_printopt   xxions(threshold=np.nan)

#print("y", y)
#print("yg", ygorrito)
