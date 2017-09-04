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

# Function to calculate Sigmoidal function
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
    # Transpose the function for easier operations
    X = X_old.transpose()
    J = 0
    grad = []
    m = len(y)
    for i in range(m):
        # Calculate sigmoidal with the sigmoidal() function
        sigmoidal_var = sigmoidal(h(X[i], theta))
        # Calculate J
        J = ((-1 * y[i] * np.log(sigmoidal_var)) - (1 - y[i] * np.log(1 - sigmoidal_var)))/m

    # Calculate grad
    for i in range(len(X_old) ):
        grad.append(sum(( sigmoidal(h(xi, theta) ) - yi) * xi[i] for xi, yi in zip(X, y) )/m)

    return(J, grad)

# This function makes the theta vector so we can make predictions
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

# Prefict function
def predice(theta,X):
    predictions = []
    X = put_ones(X)
    X = X.transpose()
    # We check in all the values of X
    for i in range(0,len(X)):
        # If the result is higher than 0.5 then is one == approved
        if(sigmoidal( h(X[i], theta)) >= 0.5):
            predictions.append(1)
        else:
            # Else is 0 == failed
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

# Function to graph
def graficaDaton(Z, y, theta):
    plt.plot(J_Historial, 'ro')
    plt.show()








#Nos deberian dar una theta normalizada y una X normalizada, Y debe ser 1 o 0
def graficaDatos(X,Y,theta):
    for _x,_y in zip(X,Y): plt.scatter( _x[0],_x[1], marker="x" if _y else 'o' ) #Puntos X/Y
    #Nos dieron datos crudos pero nuestras thetas estan normalizadas
    #nX = normalizacionDeCaracteristicas(X)[0]
    x1_min = np.amin(X,axis=0)[0]
    x1_max = np.amax(X,axis=0)[0]
    #Dos valores es suficiente puesto que es un recta
    xs = [x1_min, x1_max]
    #0.5 es la brecha de cuando pasa o no pasa el examen
    f = lambda x1,th : (0.5-th[0]-th[1]*x1)/th[2]
    #Evaluar x2 por cada x1
    plt.plot( xs  , [f(xi,theta) for xi in xs] )
    #plt.legend() # Add a legend
    plt.show()   # Show the plot
