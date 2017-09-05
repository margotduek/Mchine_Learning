from Proyecto4 import *

if __name__ == "__main__":
    x, y = parse('ex2data1.txt')
    theta = initializeTheta(x)
    x, mu, sigma = normalizacionDeCaracteristicas(x)
    funcionCosto(theta, x, y)
    theta = aprende(theta,x,y)
    costo, gradientesThetas = funcionCosto(theta,x,y)
    p = predice(theta, x)
    graficaDatos(x,y,theta)
