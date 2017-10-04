# Recibe el vector de entrada X, el de salida y, y un vector theta. Debe regresar
# la función de costo J y el gradiente grad. Es decir, debe regresar las variables [J,grad].
#def funcionCostoPerceptron(theta,X,y):

# Implementa el entrenamiento del perceptrón.
# La función debe regresar pesos del perceptrón en una matriz , done N es el número de pesos de la neurona.
# El vector y tiene etiquetas {0,1} que corresponde al valor del verdad de la función lógica para la que se está entrenando la ANN.
def entrenaPerceptron(X, y, theta):
    keep_going = 0
    while keep_going < 4:
        keep_going = 0
        for i in range(len(X)):
            error = 0
            new_y = predice(theta, X[i])
            error += y[i] - new_y
            for j in range(len(theta)):
                theta[j] += error * X[i][j]
                print ("new th ", theta[j])
            if( error == 0):
                keep_going += 1
            print ("error" , error)
            print ("--------")
    return theta


def predice(theta, X):
    for i in range(len(X)):
        net = sum(theta*x for theta,x in zip(theta,X))
        if (net > 0):
            return 1
        return 0



# Recibe un vector theta y un vector X para varios entradas lógicas. Regresa el
# vector p de predicción sobre el valor de verdad de las operaciones correspondientes.
# Si se prueba con el mismo vector X de entrenamiento debe tener una exactitud del 100% (porque es un ejemplo muy pero muy simple).
def predicePerceptron(theta, X):
    new_theta = []
    for i in range(len(X)):
        net = 0
        for j in range(len(X[i])):
            net = sum(theta*x for theta,x in zip(theta,X[i]))
        if (net > 0):
            new_theta.append(1)
        else:
            new_theta.append(0)
    print(new_theta)
    return new_theta



# def funcionCostoAdaline(theta,X,y):
# def entrenaAdaline(X, y, theta):
# def prediceAdaline(theta, X):


X = [[1,0,0], [1,0,1], [1,1,0], [1,1,1]]
y = [0,1,1,1]
theta = [1.5, 0.5, 1.5]

t = entrenaPerceptron(X, y, theta)
n_t = predicePerceptron(t, X)
