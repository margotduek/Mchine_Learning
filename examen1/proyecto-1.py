import time
import matplotlib.pyplot as plt
import numpy as np

def main():
    # We read the file
    x, y = parse("ex1data1.txt")
    # We calculate Theta and we save it in a varible
    theta = gadienteDescendente(x, y, [0,0])
    print "theta is ", theta
    print "The cost is ", calculaCosto(x, y, theta)
    graficaDatos(x, y, theta)

# Function to parse the data from the file and save it on vectors
def parse(filename):
    # Open the file
    text_file = open(filename, "r")

    lines = text_file.readlines()
    X = []
    y = []

    for line in lines:
        # split the data where it finds a coma
        X_raw, y_raw = line.split(',')
        # Save the data in X and y
        X.append(float(X_raw))
        y.append(float(y_raw))
    # Close the file
    text_file.close()

    return X, y

def graficaDatos(X,y,theta):
    # Calculate m
    m = len(X)
    # new_y is the linear regression
    new_y = []
    # Extract theta_0 and theta_1 from theta
    theta_0, theta_1 = theta
    # print each point
    plt.plot(X, y, 'ro')
    # Calculate the linear regression
    for i in range(m):
        new_y.append((theta_1 * X[i]) + theta_0)
    # Print the linear regression
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
            # calculate temp_theta_0 and temp_theta_1
            temp_theta_0 += (theta_0 + theta_1 * X[j] ) - y[j]
            temp_theta_1 += ((theta_0 + theta_1 * X[j] ) - y[j])* X[j]

        # calculating theta_0 and theta_1
        theta_0 = (theta_0 - (alfa* temp_theta_0)/m)
        theta_1 = (theta_1 - (alfa* temp_theta_1)/m)
        alfa *= 0.01;


    return (theta_0, theta_1)

def calculaCosto(X,y,theta):
    theta_0, theta_1 = theta
    err = 0.0
    m = len(X)
    # Calculate the error
    for i in range(m):
        err += np.sqrt((y[i] - (theta_0 * X[i] + theta_1))**2)
    # return err/2m acording to the formula
    return (err/2*len(X))


if __name__ == '__main__':
    main()
