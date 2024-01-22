import numpy as np
import matplotlib.pyplot as plt
from random import random

# x = np.array([1, 3, 4, 6, 3, 6, 7, 2])
# y = np.array([5, 2, 5, 7, 1, 4, 3, 5])

x = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5])
y = np.array([4, 3, 2.5, 1, 2, 3.5, 6, 4, 7, 1.5, 5, 2.5, 5.5, 3, 8, 7, 7.5, 6, 8.5, 9.5])

nr_iter = 10

def myFunc(x, b0, b1):
    return b0 + b1*x



def mse(b0,b1):
    # aplic (6)
    sumY = 0
    for i in range(len(y)):
        sumY += ((y[i] - (b0 + b1*x[i]))**2)
    return sumY/ len(y)
        

def analitic():
    
    # aplic (5)
    
    
    # sumXY = (xi - med(x))*(yi-med(y))
    sumXY = 0
    #sumX = (xi - med(x))
    sumX = 0

    for i in range(len(x)):
        sumX += ((x[i] - np.mean(x))**2)
        sumXY += (x[i] - np.mean(x)) * (y[i] - np.mean(y))
    
        
    b1 = sumXY / (sumX)
    b0 = np.mean(y) - b1*np.mean(x)
    
    return b0,b1


def sse(b0, b1):
    # aplic (7)
    suma = 0
    for i in range(len(x)):
        suma += ((y[i] -  (b0 + b1*x[i]))**2)
        
    suma = suma/2
    return suma

        
def gradient_desc(nr_iter, alpha, delta):
    b0 = random()
    b1 = random()
  
    i = 0
    while i < nr_iter:
        grad_b0 = (sse(b0 + delta, b1) - sse(b0, b1))/ (delta + 1.0)
        grad_b1 = (sse(b0, b1 + delta) - sse(b0, b1))/ (delta + 1.0)
        
        b0 -= alpha*grad_b0
        b1 -= alpha*grad_b1
            
        i+=1
    return b0,b1




b0, b1 = analitic()
print("\tSolutia Analitica")
print(f" b0 = {round(b0, 3)}  b1 = {round(b1, 3)}")
print(f" MSE = {round(mse(b0,b1),3)}")
plt.scatter(x, y)
xplot = np.arange(min(x), max(x), 0.01)
plt.plot(xplot, myFunc(xplot,b0, b1)) 
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()



print("----------------------------")
print("\n\tMetoda gradientului descendent")
b0,b1 = gradient_desc(1000,0.2,0.01)
print(f" b0 = {round(b0, 3)}  b1 = {round(b1, 3)}")
plt.scatter(x, y)
xplot = np.arange(min(x), max(x), 0.01)
plt.plot(xplot, myFunc(xplot,b0, b1)) 
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()
