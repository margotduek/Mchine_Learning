import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv


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

# Function to normalize data in y
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

# Function to initialize the theta matrix in 0.0 so it can be float
def initializeTheta(X):
    theta = []
    for i in range(len(x) + 1):
        theta.append(0.0)
    return np.array(theta)

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

# Function to calculate the hipothesis
def h(X,theta):
    return theta.transpose().dot(X)

# Function to calculate the thetas
def gadienteDescendenteMultivariable(X_old, y, theta, alfa=0.1, iteraciones=100):
    J_Historial = []
    X = put_ones(X_old)

    Xtrans = X.transpose()

    vectors = len(X)
    m = len(X[0])

    for i in range(iteraciones):
        # We generate a copy of the original Theta
        temp_theta = theta.copy()
        for j in range(vectors):
            # Calculation of theta saving it into a temporary array
            temp_theta[j] = theta[j] - (alfa/m) * sum( ( h(xi, theta ) - yi) * xi[j] for (xi,yi) in zip(Xtrans,y) )

        # saving the array on each iteration
        theta = temp_theta

        # Calculate the cost on every iteration
        J_Historial.append(calculaCosto(Xtrans,y,theta))

    return J_Historial, theta

# Function to calculate the error
def calculaCosto(X,y,theta):
    err = 0.0
    m = len(X)
    for (xi,yi) in zip(X,y):
        err += (h(xi, theta) - yi)**2
    return (err/2*m)

# Function to graph
def graficaError(J_Historial):
    plt.plot(J_Historial, 'ro')
    plt.show()

# Function to calculate theta woth the normal equation
def ecuacionNormal(X_old,y):
    X = put_ones(X_old)
    y = np.array(y)
    return  (inv(X.dot(X.transpose()))).dot(X.dot(y.transpose()))

# Function to predic the price of a house
def predicePrecio(X,theta):
    X = np.array(X)
    X = X.transpose()
    return h( np.append([1], X) ,theta)

x, y_old = parse("data.txt")
normalized_X, mu, sigma = normalizacionDeCaracteristicas(x)
y = normalizacionDeCaracteristicasy(y_old)
theta = initializeTheta(x)
historial, theta = gadienteDescendenteMultivariable(normalized_X, y, theta)
#graficaError(historial)
thetas = ecuacionNormal(x, y_old)
print 'precio ',  predicePrecio([1416,2], thetas)
