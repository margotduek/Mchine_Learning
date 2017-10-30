

def entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y):
    #Numero de capas
    L = 2
    #Capa actual
    l = 0
    for i in range(2):
        for i in range(100):
            z = np.dot(nn_params, X) + b
            A = 1.0 / (1.0 + (np.exp(-z)))
            dz = A - y
            dw = ((1/m)*np.dot(dz, X.transpose()))

            db = (1/m)*np.sum(dz)

            nn_params = nn_params - (alpha * dw)
            b = b - (alpha * db)

def sigmoidalGradiente(z):

def randInicializacionPesos(L_in, L_out):

def prediceRNYaEntrenada(X,W1,b1,W2,b2):
