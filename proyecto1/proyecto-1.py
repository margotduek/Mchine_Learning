import time
import matplotlib.pyplot as plt
import numpy as np

def main():
    x, y = parse("ex1data1.txt")
    theta = gadienteDescendente(x, y, 0)
    print "theta is ", theta
    print "The cost is ", calculaCosto(x, y, theta)
    graficaDatos(x, y, theta)
    theta_0, theta_1 = theta
    print ((theta_1 * 3.5) + theta_0)

def parse(filename):
    text_file = open(filename, "r")

    lines = text_file.readlines()
    X = []
    y = []

    for line in lines:
        X_raw, y_raw = line.split(',')
        X.append(float(X_raw))
        y.append(float(y_raw))
    text_file.close()

    return X, y

def graficaDatos(X,y,theta):
    m = len(X)
    new_y = []
    theta_0, theta_1 = theta
    plt.plot(X, y, 'ro')
    for i in range(m):
        new_y.append((theta_1 * X[i]) + theta_0)
    plt.plot(X,new_y)


    plt.show()

def gadienteDescendente(X, y, theta, alfa=0.01, iteraciones=1500):
    m = len(X)
    theta_0 = 0
    theta_1 = 0

    for i in range(iteraciones):
        temp_theta_0 = 0
        temp_theta_1 = 0
        for j in range(m):
            temp_theta_0 += (theta_0 + theta_1 * X[j] ) - y[j]
            temp_theta_1 += ((theta_0 + theta_1 * X[j] ) - y[j])* X[j]

        theta_0 = (theta_0 - (alfa* temp_theta_0)/m)
        theta_1 = (theta_1 - (alfa* temp_theta_1)/m)


    return (theta_0, theta_1)

def calculaCosto(X,y,theta):
    theta_0, theta_1 = theta
    err = 0.0
    m = len(X)
    for i in range(m):
        err += np.sqrt((y[i] - (theta_0 * X[i] + theta_1))**2)
    return (err/2*len(X))


if __name__ == '__main__':
    main()
