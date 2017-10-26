

class Neuronal_Network(object):
    def __init__(self):
        # hyperparameters
        self.input_layer_size = 2
        self.output_layer_size = 1
        self.hidden_layer_size = 3

        # Weights
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)


    def forward(self, X):
        # propagate infputs throug the network
        self.z2 = np.dot(X, self.W1)
        slef.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yhat = self.sigmoid(self.z3)
        return yhat

    def simoid(z):
        #sigmoidal activation function
        return 1/(1+np.exp(-z))

    def cost_function_prime(self, X, y):
        # compute derivative with respect of w1 and w2
        self.yhat = self.forward(X)

        delta3 = np.multiply(-(y-self.yhat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def sigmoidPrime(self, z):
        # gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    def setParams(self, params):
        # Set w1 and w2 using a simple parameter vector
        W1_start = 0
        W1_end = self.hidden_layer_size * slef.input_layer_size
        self.W1 = np.reshape(params[W1_start:W1_end], (self.input_layer_size, slef.hidden_layer_size))
        W2_end = W1_end + self.hidden_layer_size*slef.output_layer_size
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hidden_layer_size, self.output_layer_size))

    def getParams(self):
        #get W1 and w2 rolled into vector
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.cost_function_prime(X, y):
            return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)

            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #compute numerical gradiente
            numgrad[p] = (loss2 - loss1) / (2*e)

            perturb[p] = 0

        N.setParams(paramsInitial)
        return numgrad
