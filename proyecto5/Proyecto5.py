import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import plot, ylim

def costo(y, b, X, nn_params, activacion):
    J = 0
    m = X.shape[0]
    X = X.transpose()
    y = y.transpose()
    for i in range(m):
        if(activacion == "lineal"):
            J += (y[i] - np.dot(nn_params.T, X[i]) + b)**2
        else:
            sig = 1/(1 + np.exp(-(np.dot(nn_params, X[i]) + b)))
            J += -y[i]*np.log(sig)-(1-y[i])*np.log(1-sig)
    J /= m
    return J

def bpnUnaNeurona (nn_params, input_layer_size, X, y, alpha, activacion):
    b = randInicializaPesos(X.shape[1])[0]
    m = X.shape[1]
    err = []
    dw = np.zeros(X.shape[1])
    for i in range(1000):
        z = np.dot(nn_params, X) + b
        if( activacion == "lineal"):
            A = z
        else:
            A = 1.0 / (1.0 + (np.exp(-z)))
        dz = A - y
        dw = ((1/m)*np.dot(dz, X.transpose()))

        db = (1/m)*np.sum(dz)

        nn_params = nn_params - (alpha * dw)
        b -= alpha * db
    err.append(costo(y, b, X, nn_params, activacion))
    return b, nn_params, err

def sigmoidGradiente(z):
    sigmoidal = 1.0 / (1.0 + (np.exp(-z)))
    return sigmoidal * (1 - sigmoidal)

def linealGradiante(z):
    return 1

def randInicializaPesos(L_in):
    pesos = np.zeros(L_in)
    for i in range(L_in):
        pesos[i] = random.uniform(-0.12, 0.12)
    return pesos

def prediceRNYaEntrenada(X, nn_params, b, activacion):
    Z = np.dot(nn_params, X) + b
    if (activacion == "sigmoidal"):
        A = 1/(1 + np.exp(-Z))

        # Round numbers
        A = A.transpose()
        for i in range(len(A)):
            A[i] = is_one(A[i])
        A = A.transpose()
    elif (activacion == "lineal"):
        A = lineal(Z)
    return A

def is_one(A):
    return np.where(A < 0.5, 0,  1)

def graficar_error(errors):
    plot_errors = np.zeros(len(errors))
    for i in range(len(errors)):
        plot_errors[i] = errors[i]
    plt.plot(plot_errors)
    plt.ylabel('Error')
    plt.show()




def main():
    X = np.array([[0,0], [0,1], [1,0], [1,1]]).transpose()
    Y = np.array([[0], [0], [0], [1]]).transpose()

    pesos = randInicializaPesos(X.shape[0])

    b, nn_params_sig, err = bpnUnaNeurona(pesos, X.shape[0], X, Y, .6, 'sigmoidal')

    print("y gorrito: ", prediceRNYaEntrenada(X, nn_params_sig, b, 'sigmoidal'))

    return

if __name__ == "__main__":
    main()
