import numpy as np
import matplotlib.pyplot as plt

def f1(x): # functia care trebuie minimizata
    return x**4 - 7* x**3 + 14 * x**2 - 8 *x

def f1Grad(x): # derivata
    return 4 * x**3 - 21 * x**2 +28*x -8

f1Limits = [-0.2, 4.4] # domeniul functiei
    


if __name__ == '__main__':
    
    f = f1
    fGrad = f1Grad
    xMin, xMax = f1Limits
    
    ### gradientul descendent
    
    x = xMin+0.1 # valoarea initiala a lui x
    alpha = 0.025 # rata de invatare 0.015
    nrIter = 30
    minims = {}
    # fmin =  -6.9141
    fmin =  None
    
    while x < xMax:
        if f(x) > f(x+0.1):
        
            xGD = np.array([x]) # valorile obtinute pe parcursul iteratiilor
            
            for i in range(nrIter):
                x = x - alpha * fGrad(x)
                xGD = np.append(xGD, x)
                print('Iter {0}: x = {1} , f(x) = {2}'.format(i+1, x, round(f(x),4)))
                # ... aici ar trebui adaugat un criteriu de convergenta
                if fmin is not  None and round(f(x),4) == fmin:
                    break
                
                if fmin is None and i> (nrIter-8):
                    if round(f(xGD[i-1]),4) == round(f(xGD[i]),4) and round(f(xGD[i-2]),4) == round(f(xGD[i - 1]),4):
                        break
                    
            
        else:
            if minims and f(list(minims)[-1]) > f(x): 
                minims[x] = xGD  
            if not minims:
                minims[x] = xGD  
            x = x+0.1
                
   
    # print(minims)
    print("--------------------------------------------")
    print(' minim local: f(x) = {0}, x={1}'.format( round(f(list(minims)[-2]),4) , round(list(minims)[-2],3) ))
    print(' minim global: f(x) = {0}, x={1}'.format( round(f(list(minims)[-1]),4) , round(list(minims)[-1],3)))
    xGD = None
    fx = None
    for xi in minims:
        if xGD is not None:
            if f(xi) < fx:
                xGD=minims[xi]
                fx = f(xi)
        if  xGD is None: 
            xGD=minims[xi]
            fx = f(xi)
            
        # print(minims[xi])
        
            
            
    xGD = np.array(xGD)
    # print(xGD)
        
        
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
        
        
    




