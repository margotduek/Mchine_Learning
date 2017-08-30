import time
import matplotlib.pyplot as plt
import numpy as np



# def main():
#     x, y = parse("ex1data1.txt")
#     theta = gadienteDescendente(x, y, 0)
#     print "theta is ", theta
#     print "The cost is ", calculaCosto(x, y, theta)
#     graficaDatos(x, y, theta)
#     theta_0, theta_1 = theta
#     print ((theta_1 * 3.5) + theta_0)

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

    return normalized, mu, sigma


def gadienteDescendenteMultivariable(X_old, y, theta, alfa=0.1, iteraciones=100):
    X = []
    x = []
    for i in range(len(X_old[0])):
        x.append(1)
    X.append(x)

    for i in range(len(X_old)):
        x = []
        for j in range(len(X_old[i])):
            x.append(X_old[i][j])
        X.append(x)

    vectors = len(X)
    thetas = []
    m = len(X[0])

    for i in range(iteraciones):
        for j in range(vectors):
            temp_theta = 0
            for k in range(m):
                temp_theta += ( ( (theta[j] * X[j][k]) + (theta[j+1] * X[j][k]) ) - y[k] ) * X[j][k]

            theta[j] = (temp_theta - (alfa* temp_theta)/m)

    return (theta_0, theta_1)

    
x, y = parse("data.txt")
normalizacionDeCaracteristicas(x)
gadienteDescendenteMultivariable(x, y, 0)



# def ecuacionNormal(X,y):
#
# def predicePrecio(X,theta):
#
# def graficaError(J_Historial):
#     m = len(X)
#     new_y = []
#     theta_0, theta_1 = theta
#     plt.plot(X, y, 'ro')
#     for i in range(m):
#         new_y.append((theta_1 * X[i]) + theta_0)
#     plt.plot(X,new_y)
#     plt.show()
#

#
# def calculaCosto(X,y,theta):
#     theta_0, theta_1 = theta
#     err = 0.0
#     m = len(X)
#     for i in range(m):
#         err += np.sqrt((y[i] - (theta_0 * X[i] + theta_1))**2)
#     return (err/2*len(X))
#
#
# if __name__ == '__main__':
#     main()
