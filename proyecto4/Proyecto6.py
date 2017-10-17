
import matplotlib.pyplot as plt
import numpy as np


def grafica(e):
    plt.plot( e )
    plt.show()

#funcion para agregarle la capa de unos a la matriz
def agregaUnos(X):
    new__x = []
    for i in range(len(X)):
        new_x = []
        new_x.append(1)
        for j in range(len(X[i])):
            new_x.append(X[i][j])
        new__x.append(new_x)
    return new__x

# Implementa el entrenamiento del perceptrón.
# La función debe regresar pesos del perceptrón en una matriz , done N es el número de pesos de la neurona.
# El vector y tiene etiquetas {0,1} que corresponde al valor del verdad de la función lógica para la que se está entrenando la ANN.
def entrenaPerceptron(oX, y, theta):
    X = agregaUnos(oX)
    keep_going = 0
    error_tot = []
    # revisar que tengamos por lo menos un error para seguir revisando
    while keep_going < 4:
        keep_going = 0
        batch_error = 0
        for i in range(len(X)):
            error = 0
            #predice y
            new_y = predice(theta, X[i])
            # revisamos la y que tenemos menos la predecida para ver el error
            error += y[i] - new_y
            #calculamos theta nuevo
            for j in range(len(theta)):
                theta[j] += error * X[i][j]
            if( error == 0):
                keep_going += 1
            batch_error += abs(error)
        # agregamos el error total del batch
        error_tot.append(batch_error)
    return theta, error_tot

# Predice la nueva y
def predice(theta, X):
    for i in range(len(X)):
        net = sum(theta*x for theta,x in zip(theta,X))
        if (net > 0):
            return 1
        return 0



# Recibe un vector theta y un vector X para varios entradas lógicas. Regresa el
# vector p de predicción sobre el valor de verdad de las operaciones correspondientes.
# Si se prueba con el mismo vector X de entrenamiento debe tener una exactitud del 100% (porque es un ejemplo muy pero muy simple).
def predicePerceptron(theta, oX):
    X = agregaUnos(oX)
    new_theta = []
    for i in range(len(X)):
        net = 0
        for j in range(len(X[i])):
            net = sum(theta*x for theta,x in zip(theta,X[i]))
        if (net > 0):
            new_theta.append(1)
        else:
            new_theta.append(0)
    return new_theta


# Predice la nueva y
def predice_adaline(theta, X):
    for i in range(len(X)):
        net = sum(theta*x for theta,x in zip(theta,X))
        if (net > .5):
            return 1
        return 0

# def funcionCostoAdaline(theta,X,y):
def entrenaAdaline(oX, y, theta):
    X = agregaUnos(oX)
    total_error = 10
    error_tot = []
    # revisar que tengamos por lo menos un error para seguir revisando
    while total_error > 0.01:
        new__y = []
        for i in range(len(X)):
            error = 0
            #predice y
            new_y = predice_adaline(theta, X[i])
            new__y.append(new_y)
            # revisamos la y que tenemos menos la predecida para ver el error
            error += y[i] - new_y
            #calculamos theta nuevo
            for j in range(len(theta)):
                theta[j] += error * X[i][j]

        total_error = rmse(np.array(new__y), np.array(y))
        # agregamos el error total del batch
        error_tot.append(total_error)
    return theta, error_tot

# Calculamos el rmse
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def prediceAdaline(theta, oX):
    X = agregaUnos(oX)
    new_theta = []
    for i in range(len(X)):
        net = 0
        for j in range(len(X[i])):
            net = sum(theta*x for theta,x in zip(theta,X[i]))
        if (net > .5):
            new_theta.append(1)
        else:
            new_theta.append(0)
    return new_theta
