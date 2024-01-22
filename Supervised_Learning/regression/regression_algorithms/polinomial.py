import numpy as np
import matplotlib.pyplot as plt
x = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5])
y = np.array([4, 3, 2.5, 1, 2, 3.5, 6, 4, 7, 1.5, 5, 2.5, 5.5, 3, 8, 7, 7.5, 6, 8.5, 9.5])



def myFunc(beta, x):
    y = 0
    for power, b in enumerate(beta):
        y += b * (x ** power)
    return y


# m = grad

def polinom(grad):
    global x,y
    rows = len(x)
    cols = grad + 1
    matX =[]
    
    for row in range(rows):
        xLine = []
        for col in range(cols):
            xLine.append(x[row] ** col)
        matX.append(xLine)
        
    matX = np.array(matX)
    beta = np.dot(np.dot(np.linalg.pinv(np.dot(matX.transpose(), matX)), matX.transpose()), y)
    
    
    return beta
    
 
def mse(beta):
    global x,y
    yb = myFunc(beta, x)
    
    sumY = 0
    for i in range(len(y)):
        sumY += (yb[i] - y[i])**2

    return sumY/ len(y)

grad = 2
beta = polinom(grad)

print(f"grad: {grad} MSE = {round(mse(beta),3)}")

plt.scatter(x, y)
xplot = np.arange(min(x), max(x), 0.01)
plt.plot(xplot, myFunc(beta, xplot))
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()




