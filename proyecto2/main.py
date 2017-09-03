from Proyecto3 import *

if __name__ == "__main__":
    x, y_old = parse("data.txt")
    normalized_X, mu, sigma = normalizacionDeCaracteristicas(x)
    y = normalizacionDeCaracteristicasy(y_old)
    theta = initializeTheta(x)
    historial, theta = gadienteDescendenteMultivariable(normalized_X, y, theta)
    #graficaError(historial)
    thetas = ecuacionNormal(x, y_old)
    print 'precio ',  predicePrecio([1416,2], thetas)   
