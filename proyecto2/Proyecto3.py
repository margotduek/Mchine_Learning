import time
import matplotlib.pyplot as plt
import numpy as np

def parse(filename):
    text_file = open(filename, "r")

    lines = text_file.readlines()
    X = []
    y = []
    new_x = []

    for i in lines:
        X.append([])

    for i, line in enumerate(lines):
        x_vars = line.split(',')
        for x_var in x_vars[:-1]:
            X[i].append(float(x_var))
        y.append(float(x_vars[-1]))
    text_file.close()
    new_x = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]
    return np.array(new_x), np.array(y)


def normalizacionDeCaracteristicas(X):
    m = len(X)
    mu = []
    sigma = []
    for i in range(m):
        mu.append(np.mean(X[i]))

    for i in range(m):
        sigma.append(X[i].std())

    normalized = []
    temp = 0
    for i in range(m):
        normalized_temp = []
        for j in range(len(X[i])):
            temp = ((X[i][j] - mu[i]))/sigma[i]
            normalized_temp.append(temp)
        normalized.append(normalized_temp)

    return np.array(normalized), mu, sigma

def normalizacionDeCaracteristicasy(y):
    m = len(y)
    mu = np.mean(y)
    sigma = y.std()

    normalized = []
    temp = 0
    normalized_temp = []
    for j in range(len(y)):
        temp = ((y[j] - mu))/sigma
        normalized.append(temp)

    return np.array(normalized)


def initializeTheta(X):
    theta = []
    for i in range(len(x) + 1):
        theta.append(0.0)
    return np.array(theta)

def put_ones(X_old):
    x = []
    X = []
    for i in range(len(X_old[0])):
        x.append(1)
    X.append(x)

    for i in range(len(X_old)):
        x = []
        for j in range(len(X_old[i])):
            x.append(X_old[i][j])
        X.append(x)
    X = np.array(X)
    return X

def h(X,theta):
    return theta.transpose().dot(X)

def gadienteDescendenteMultivariable(X_old, y, theta, alfa=0.1, iteraciones=100):
    J_Historial = []
    X = put_ones(X_old)

    Xtrans = X.transpose()

    vectors = len(X)
    m = len(X[0])

    for i in range(iteraciones):
        temp_theta = theta.copy() #Thetas temporales
        for j in range(vectors):
            temp_theta[j] = theta[j] - (alfa/m) * sum( ( h(xi, theta ) - yi) * xi[j] for (xi,yi) in zip(Xtrans,y) )

        theta = temp_theta
        J_Historial.append(calculaCosto(Xtrans,y,theta))

    return J_Historial, theta


def calculaCosto(X,y,theta):
    err = 0.0
    m = len(X)
    for (xi,yi) in zip(X,y):
        err += (h(xi, theta) - yi)**2
    return (err/2*len(X))

def graficaError(J_Historial):
    plt.plot(J_Historial, 'ro')
    plt.show()


x, y_old = parse("data.txt")
normalized_X, mu, sigma = normalizacionDeCaracteristicas(x)
y = normalizacionDeCaracteristicasy(y_old)
theta = initializeTheta(x)
historial, theta = gadienteDescendenteMultivariable(normalized_X, y, theta)
graficaError(historial)

def ecuacionNormal(X_old,y):
    X = put_ones(X_old)
    return  (inv(X.transpose().dot(X))).dot(X.transpose().dot(Y))


def predicePrecio(X,theta):
    return h( np.append([1],X) ,theta)
