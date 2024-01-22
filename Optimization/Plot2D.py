import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import random

def f1(x, y):
    return (1-(x**2+y**3))*np.exp(-(x**2+y**2)/2)

f1Limits = [[-3, 3],[-3, 3]]

if __name__ == '__main__':

    f = f1
    xMin, xMax = f1Limits[0]
    yMin, yMax = f1Limits[1]

    plt.rc('font', size=15)
    lineWidth = 2
    dotSize = 12

    fstep = 0.2 # pasul folosit pentru afisare (valori mai mici = rezolutie mai mare)
    x = np.arange(xMin, xMax, fstep)
    y = np.arange(yMin, yMax, fstep)
    x, y = np.meshgrid(x, y)
    z = f(x, y)
    
    ## afisare sub forma unei imagini 2D
    plt.imshow(np.flip(z, 0), cmap = plt.get_cmap('gray'), extent=(xMin, xMax, yMin, yMax))
    
    ## afisarea valorilor gradientului descendent, presupunand ca acestea se retin in xGD, yGD
    #plt.plot(xGD, yGD, '--o', color='orange', linewidth=lineWidth, markerSize=dotSize)
    #plt.plot(xGD[0], yGD[0], 'o', color='blue', markersize=dotSize)
    #plt.plot(xGD[-1], yGD[-1], 'o', color='red', markersize=dotSize)

    plt.xlim((xMin, xMax))
    plt.ylim((yMin, yMax))
    plt.tight_layout()
    
    ### afisare sub forma unei suprafete intr-o scena 3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, 
                    rstride = 1, cstride=1, 
                    cmap=plt.get_cmap('gray'), 
                    linewidth=0, antialiased = True,
                    zorder=0)

    ## afisarea valorilor gradientului descendent, presupunand ca acestea se retin in xGD, yGD
    #zGD = f(xGD, yGD)
    #ax.plot(xGD, yGD, zGD, '--o', color='orange', zorder=10, linewidth=lineWidth, markersize=dotSize)
    #ax.plot([xGD[0]], [yGD[0]], [zGD[0]], 'o', color='blue', zorder=10, markersize=dotSize)
    #ax.plot([xGD[-1]], [yGD[-1]], [zGD[-1]], 'o', color='red', zorder=10, markersize=dotSize)

    ax.view_init(30, 40) # stabilirea unghiurilor din care se priveste scena 3D (in grade)
    
    ## rotirea automata a scenei 3D 
    ## (posibil doar daca matplotlib genereaza o fereastra separata)
    ## (nu functioneaza pentru ferestre 'inline')
    
    #for angle in range(0, 360):
    #    ax.view_init(30, angle)
    #    plt.draw()
    #    plt.pause(0.001)
    
    plt.tight_layout()
    plt.show()



