import numpy as np
import matplotlib.pyplot as plt
import random

# x = np.array([1, 3, 4, 6, 3, 6, 7, 2])
# y = np.array([5, 2, 5, 7, 1, 4, 3, 5])

x = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])





def sigm(x):
    return 1/(1+ np.exp(-x))


def train(epoci, alpha, ct_error):
    
    w0 = random.uniform (-0.5,0.5)
    w1 = random.uniform (-0.5,0.5)
        
    while epoci > 0:
        CE = 0
        
        grad_w0 = 0
        grad_w1 = 0
        
        
        for i in range(len(y)):
            
            CE += ( y[i] * np.log( sigm(w0 + w1*x[i]) )  + 
                               (1-y[i]) * np.log(1 - sigm(w0 + w1*x[i]) ))
            grad_w0 += ( sigm(w0 + w1*x[i]) - y[i])
            grad_w1 += (( sigm(w0 + w1*x[i]) - y[i]) * x[i])
            
        
        w0 -=  alpha * grad_w0
        w1 -=  alpha * grad_w1
        CE = - CE / len(y)
        if round(CE,4) == ct_error:
            break;
        
        epoci -= 1

    return w0, w1



w0,w1 = train(1000, 0.01, 0.001)



print(f" w0 = {round(w0, 3)}  w1 = {round(w1, 3)}")
plt.scatter(x, y)
xplot = np.arange(min(x), max(x), 0.01)
plt.plot(xplot, sigm(w0 + w1*xplot)) 
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()
