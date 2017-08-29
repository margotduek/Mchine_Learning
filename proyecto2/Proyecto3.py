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
    m = len(X[0])
    sum_0 = 0
    sum_1 = 1
    for i in range(m):
        sum_0 += X[0][i]
        sum_1 += X[1][i]
    media_0 = sum_0/m
    media_1 = sum_1/m
    normalized_0 = []
    normalized_1 = []
    print "media", media_0

    for i in range(m):
        normalized_0.append((X[0][i] - media_0)/ (np.amax(X[0]) - np.amin(X[0])))
        normalized_1.append((X[1][i] - media_1)/ (np.amax(X[1]) - np.amin(X[1])))
        print "rango", (np.amax(X[1]) - np.amin(X[1]))
        print "normalizado", normalized_0[i]

    print normalized_0, normalized_1
    return normalized_0, normalized_1


x, y = parse("data.txt")
normalizacionDeCaracteristicas(x)



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
# def gadienteDescendenteMultivariable(X, y, theta, alfa=0.1, iteraciones=100):
#     m = len(X)
#     theta_0 = 0
#     theta_1 = 0
#
#     for i in range(iteraciones):
#         temp_theta_0 = 0
#         temp_theta_1 = 0
#         for j in range(m):
#             temp_theta_0 += (theta_0 + theta_1 * X[j] ) - y[j]
#             temp_theta_1 += ((theta_0 + theta_1 * X[j] ) - y[j])* X[j]
#
#         theta_0 = (theta_0 - (alfa* temp_theta_0)/m)
#         theta_1 = (theta_1 - (alfa* temp_theta_1)/m)
#
#
#     return (theta_0, theta_1)
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
