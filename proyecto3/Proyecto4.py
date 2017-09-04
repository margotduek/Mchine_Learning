import time
import matplotlib.pyplot as plt
import numpy as np

# Function to parse the data from the txt to an array and a matrix
def parse(filename):
    text_file = open(filename, "r")
    lines = text_file.readlines()
    X = []
    y = []
    new_x = []

    # appending the data into X
    for i in lines:
        X.append([])

    for i, line in enumerate(lines):
        x_vars = line.split(',')
        for x_var in x_vars[:-1]:
            X[i].append(float(x_var))
        y.append(float(x_vars[-1]))
    text_file.close()

    # We transpose the matrix to make operations easier
    X = np.array(X)
    new_x = X.transpose()
    # We return X an y as a numpy array
    return np.array(new_x), np.array(y)

# def graficaDatos(X,y,theta):

def sigmoidal(z):
    return 1.0/(1.0 + np.e**( -z ) )

# Function to calculate the hipothesis
def h(X,theta):
    return theta.transpose().dot(X)

# Function to add an array of ones at the beguining of X
def put_ones(X_old):
    x = []
    X = []
    # We add the ones at the beguining of the array
    for i in range(len(X_old[0])):
        x.append(1)
    X.append(x)

    # We add the array after the ones
    for i in range(len(X_old)):
        x = []
        for j in range(len(X_old[i])):
            x.append(X_old[i][j])
        X.append(x)
    # make the array an numpy array
    X = np.array(X)
    return X


# Function to initialize the theta matrix in 0.0 so it can be float
def initializeTheta(X):
    theta = []
    for i in range(len(X) + 1):
        theta.append(0.0)
    return np.array(theta)


def funcionCosto(theta,X_old,y):
    X_old = put_ones(X_old)
    X = X_old.transpose()
    J = 0
    grad = []
    m = len(y)
    for i in range(m):
        sigmoidal_var = sigmoidal(h(X[i], theta))
        J = ((-1 * y[i] * np.log(sigmoidal_var)) - (1 - y[i] * np.log(1 - sigmoidal_var)))/m

    for i in range(len(X_old) ):
        grad.append(sum(( sigmoidal(h(xi, theta) ) - yi) * xi[i] for xi, yi in zip(X, y) )/m)

    return(J, grad)

def aprende(theta,X_old,y,iteraciones = 1500):
    X = put_ones(X_old)
    Xtrans = X.transpose()

    vectors = len(X)
    m = len(X[0])

    for i in range(iteraciones):
        # We generate a copy of the original Theta
        temp_theta = theta.copy()
        for j in range(vectors):
            # Calculation of theta saving it into a temporary array
            temp_theta[j] = theta[j] - ((sum( ( sigmoidal(h(xi, theta)) - yi) * xi[j] for (xi, yi) in zip(Xtrans, y)))/m)
        # saving the array on each iteration
        theta = temp_theta

    return theta

def predice(theta,X):
    predictions = []
    X = put_ones(X)
    X = X.transpose()
    for i in range(0,len(X)):
        if(sigmoidal( h(X[i], theta)) >= 0.5):
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# Function to normalize data in X
def normalizacionDeCaracteristicas(X):
    m = len(X)
    mu = []
    sigma = []

    # Calculate the mu of each of the variables in ix
    for i in range(m):
        mu.append(np.mean(X[i]))

    # Calculate the sigma of each of the variables in ix
    for i in range(m):
        sigma.append(X[i].std())

    normalized = []
    temp = 0
    for i in range(m):
        normalized_temp = []
        for j in range(len(X[i])):
            # normalize the data
            temp = ((X[i][j] - mu[i]))/sigma[i]
            # append the data into an array
            normalized_temp.append(temp)
        # append the array of the normalized data into the matrix
        normalized.append(normalized_temp)

    return np.array(normalized), mu, sigma


x, y = parse('ex2data1.txt')
theta = initializeTheta(x)
x, mu, sigma = normalizacionDeCaracteristicas(x)
funcionCosto(theta, x, y)
theta = aprende(theta,x,y)
costo, gradientesThetas = funcionCosto(theta,x,y)
p = predice(theta, x)

print p
print y
print theta
