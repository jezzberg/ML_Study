import numpy as np
import matplotlib.pyplot as plt
import numpy as np



delta = 0.1

def f1(x): # functia care trebuie minimizata
    return np.sin(np.sqrt(x))/x 

def f1Grad(x): # derivata
    return (f(x+delta) - f(x-delta)) / (2* delta)

f1Limits = [0, 40] # domeniul functiei
    

if __name__ == '__main__':
    
    f = f1
    fGrad = f1Grad
    xMin, xMax = f1Limits
    # fmin =  -0.0496
    fmin =  None
    
    ### gradientul descendent

    x = xMin+2 # valoarea initiala a lui x
    alpha = 3 # rata de invatare 
    nrIter = 4000

    xGD = np.array([x]) # valorile obtinute pe parcursul iteratiilor
    
    for i in range(nrIter):
       
        x = x - alpha * fGrad(x)
        xGD = np.append(xGD, x)
        print('Iter {0}: x = {1} , f(x) = {2}'.format(i+1, round(x,3), f(x)))
        if fmin is not  None and round(f(x),4) == fmin:
                    break
                
        if fmin is None and i> (nrIter-8):
            if round(f(xGD[i-1]),4) == round(f(xGD[i]),4) and round(f(xGD[i-2]),4) == round(f(xGD[i - 1]),4):
                break
    
    print("---------------")
    print("- minim global: f(x) = {0}, x={1}\n".format(round(f(x),4), round(x,3)))
    yGD = f(xGD)

    ### afisare grafica
    
    plt.rc('font', size=15)
    lineWidth = 2
    dotSize = 12

    LOD = 30 # nr de puncte prin care se traseaza graficul functiei
    stepSize = (xMax - xMin) / (LOD-1)
    x = np.arange(xMin, xMax + stepSize, stepSize)
    y = f(x)
    plt.plot(x, y, '-', linewidth = lineWidth) # afisarea graficului functiei
    
    ## afisarea valorilor obtinute cu gradientul descendent
    gradx = fGrad(x)
    plt.plot(xGD, yGD, '--o', color='orange', linewidth = lineWidth, markersize=dotSize)
    # primul si ultimul punct sunt evidentiate cu alte culori
    plt.plot(xGD[0], yGD[0], 'o', color='blue', markersize = dotSize)
    plt.plot(xGD[-1], yGD[-1], 'o', color='red', markersize = dotSize)

    plt.xlim((xMin, xMax))
    plt.tight_layout()
    plt.show()
    
    




