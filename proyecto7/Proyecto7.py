import random
import numpy as np
import csv
import matplotlib.pyplot as plt
from pylab import plot, ylim

# Funcion para encontrar el centroide mas cercano a cada punto
def findClosestCentroids(X, initial_centroids):
    # arreglo para cuardar el valor de la distancia entre dos puntos de cada valor de X
    cluster_size = len(initial_centroids)
    clusters = [0] * cluster_size
    # arreglo para guardar a que cluster pertenece cada putno
    cluster = []

    for i in range(len(X)):
        for j in range(cluster_size):
            # C² = a² + b²
            # Aplicamos esta formula para cada punto y cada cluster seleccionado para ver a que cluster pertenece cada numero
            clusters[j] = np.sqrt((X[i][0] - initial_centroids[j][0])**2 + (X[i][1] - initial_centroids[j][1])**2 )
        min_num = np.amin(clusters)
        # Metemos al arreglo cluster el cluster al que pertenece cada punto
        cluster.append(clusters.index(min_num))
    return cluster


# Funcion para encontrar clusters mas centricos
def computeCentroids(X, idx, K):
    # inicializamos arreglos del tamañno de k
    totsX = [0] * K
    totsY = [0] * K
    tots = np.zeros((2, K))
    Xs = [0] * K
    Ys = [0] * K
    ammount = [0] * K
    for i in range(len(X)):
        cluster = idx[i]
        # Hacemos la sumatoria de cada X y cada Y en cada cluster
        Xs[cluster] = Xs[cluster] + X[i][0]
        Ys[cluster] = Ys[cluster] + X[i][1]
        # Vemos cuantos puntos hay en cada cluster
        ammount[cluster] = ammount[cluster] + 1

    # Sacamos el promedio de cada cluster
    for j in range(K):
        totsX[j] = Xs[j]/ammount[j]
        totsY[j] = Ys[j]/ammount[j]
    tots[0] = totsX
    tots[1] = totsY
    return tots.transpose()

# Fuincion para encontrar los centroides
def kMeansInitCentroids(X, K):
    centroids = []
    permutated = np.random.permutation(X)
    for i in range(K):
        centroids.append(permutated[i].tolist())
    return centroids

# Leemos el archivo y guardamos los datos en una matriz
def read_file(filename):
    text_file = open(filename, "r")

    lines = text_file.readlines()
    X = []
    y = []
    data = []

    for line in lines:
        # split the data where it finds a space
        X_raw, y_raw = line.split(' ')
        # Save the data in X and y
        X.append(float(X_raw))
        y.append(float(y_raw))
    # Close the file
    text_file.close()
    data.append(X)
    data.append(y)
    data = np.array(data)
    return data.transpose()

# Iteramos y recalculamos que cluster es el que conviene mas
def runkMeans(X, centroids, max_iters, true):
    idx = findClosestCentroids(X, centroids)
    old_centroids = []
    for ii in range(max_iters):
        for i in range(len(X)):
            if(idx[i] == 0):
                plt.scatter(X[i][0], X[i][1], color = 'r')
            if(idx[i] == 1):
                plt.scatter(X[i][0], X[i][1], color = 'g')
            if(idx[i] == 2):
                plt.scatter(X[i][0], X[i][1], color = 'b')
        for i in range(len(centroids)):
            plt.scatter(centroids[i][0], centroids[i][1], color = 'y', s = 130, marker = 'x', linewidth = '7')
            if(true):
                for j in range(len(old_centroids)):
                    plt.scatter(old_centroids[j][i][0], old_centroids[j][i][1], color = 2, s = 70, marker = 'x', linewidth = '4')
                    if(len(centroids) > 0):
                        plt.plot([old_centroids[j-1][i][0],old_centroids[j][i][0]], [old_centroids[j - 1][i][1], old_centroids[j][i][1]])


        old_centroids.append(centroids)
        plt.show()
        centroids = computeCentroids(X, idx, 3)
        idx = findClosestCentroids(X, centroids)



x = read_file("ex7data2.txt")
initial_centroids = kMeansInitCentroids(x, 3)
# idx = findClosestCentroids(x, initial_centroids)
# new_centroids = computeCentroids(x, idx, 3)
runkMeans(x, initial_centroids, 10, True)
