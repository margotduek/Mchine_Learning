import random
import numpy as np
import csv
import matplotlib.pyplot as plt
from pylab import plot, ylim


def findClosestCentroids(X, initial_centroids):
    # set_to_zero(initial_centroids)
    clusters = []
    cluster_size = 0
    cluster = []
    for i in range(len(initial_centroids)):
        temp = "__" + str(i)
        clusters.append(temp)
        cluster_size = cluster_size + 1

    for i in range(len(X)):
        for j in range(cluster_size):
            clusters[j] = np.sqrt((X[i][0] - initial_centroids[j][0])**2 + (X[i][1] - initial_centroids[j][1])**2 )
        min_num = np.amin(clusters)
        cluster.append(clusters.index(min_num))
    return cluster


def computeCentroids(X, idx, K):
    totsX = [0] * K
    totsY = [0] * K
    tots = np.zeros((2, K))
    Xs = [0] * K
    Ys = [0] * K
    ammount = [0] * K
    for i in range(len(X)):
        cluster = idx[i]
        Xs[cluster] = Xs[cluster] + X[i][0]
        Ys[cluster] = Ys[cluster] + X[i][1]
        ammount[cluster] = ammount[cluster] + 1
        # print(Xs[cluster], cluster, ammount[cluster])

    for j in range(K):
        totsX[j] = Xs[j]/ammount[j]
        totsY[j] = Ys[j]/ammount[j]
        print(totsX[j])
    tots[0] = totsX
    tots[1] = totsY


    print(tots.transpose())

    #         Xs[j]
    #         temp = temp = "X" + str(j)
    #         temp = temp + X[i][0]
    # print(X0)
    #         X0 = X0 + X[i][0]
    #         Y0 = Y0 + X[i][1]
    #         tot_0 = tot_0 + 1
    #
    #     else if(idx[i] == 1):
    #         X1 = X1 + X[i][0]
    #         Y1 = Y1 + X[i][1]
    #         tot_1 = tot_1 + 1
    #     else if(idx[i] == 0):
    #         X2 = X2 + X[i][0]
    #         Y2 = Y2 + X[i][1]
    #         tot_2 = tot_2 + 1
    # x0 = X0/tot_0
    # y0 = Y0/tot_0
    # x1 = X1/tot_1
    # y1 = Y1/tot_1
    # x2 = X2/tot_2
    # y2 = Y2/tot_2



# def runkMeans(X, initial_centroids, max_iters, true):
    # Es el pseudocódigo Azul

def kMeansInitCentroids(X, K):
    # permutated = np.array(X)
    centroids = []
    permutated = np.random.permutation(X)
    for i in range(K):
        centroids.append(permutated[i].tolist())
    return centroids

def read_file(filename):
    # Open the file
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


x = read_file("ex7data2.txt")
initial_centroids = kMeansInitCentroids(x, 3)
idx = findClosestCentroids(x, initial_centroids)
computeCentroids(x, idx, 3)
