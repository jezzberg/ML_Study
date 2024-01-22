import numpy as np
import matplotlib.pyplot as plt

def f1(x,y): # functia care trebuie minimizata
    return x**4 + 2*(x**2)*y - 21*(x**2) + 2*x*(y**2) -14*x +(y**4) - 13*(y**2) - 22*y + 170

def fxGrad(x,y): # derivata
    return 4*(x**3) + 4*x*y - 42*x +2*(y**2) -14
def fyGrad(x,y): # derivata
    return 2*(x**2) + 4*x*y + 4*(y**3) -26*y -22

f1Limits = [[-4, 4],[-4, 4]]    


if __name__ == '__main__':
    
    f = f1
    fGradx = fxGrad
    fGrady = fyGrad
    
    xMin, xMax = f1Limits[0]
    yMin, yMax = f1Limits[1]
    
    ### gradientul descendent
    
    
    
    alpha = 0.001 # rata de invatare 
    nrIter = 300
    minims = {}
    # fmin =  -6.9141
   
    
    fXmin =  0
    fYmin =  0
    
    globalsMin = []
    
    for xstep in range(int(xMin/2),xMax,xMax):
        for ystep in range(int(yMin/2),yMax,yMax):
        
            x = xstep 
            y = ystep
            
            xGD = np.array([x]) # valorile obtinute pe parcursul iteratiilor
            yGD = np.array([y])
            
            for i in range(nrIter):
                x = x - alpha * fGradx(x,y)
                y = y - alpha * fyGrad(x,y)
                xGD = np.append(xGD, x)
                yGD = np.append(yGD, y)
                print('Iter {0}: x = {1} ,y = {2}, f = {3} '.format(i+1, 
                                                    round(x,4), round(y,4), round(f(x,y),4)))
                if round(f(x,y),4) == 0:
                            break
                        
                if i> (nrIter-8):
                    if round(f(xGD[i-1],yGD[i-1]),4) == round(f(xGD[i],yGD[i]),4) and round(f(xGD[i-2],yGD[i-2]),4) == round(f(xGD[i-1],yGD[i-1]),4):
                        break
               
        
                # ... aici ar trebui adaugat un criteriu de convergenta
            print()
            print("---------------\t\t x = {0} \t y = {1}".format(x,y))
            print()
            minim = (x,y)
            globalsMin.append(minim)
            
            
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
            plt.plot(xGD, yGD, '--o', color='orange', linewidth=lineWidth, markerSize=dotSize)
            plt.plot(xGD[0], yGD[0], 'o', color='blue', markersize=dotSize)
            plt.plot(xGD[-1], yGD[-1], 'o', color='red', markersize=dotSize)
        
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
            zGD = f(xGD, yGD)
            ax.plot(xGD, yGD, zGD, '--o', color='orange', zorder=10, linewidth=lineWidth, markersize=dotSize)
            ax.plot([xGD[0]], [yGD[0]], [zGD[0]], 'o', color='blue', zorder=10, markersize=dotSize)
            ax.plot([xGD[-1]], [yGD[-1]], [zGD[-1]], 'o', color='red', zorder=10, markersize=dotSize)
        
            ax.view_init(30, 40) # stabilirea unghiurilor din care se priveste scena 3D (in grade)
            
            ## rotirea automata a scenei 3D 
            ## (posibil doar daca matplotlib genereaza o fereastra separata)
            ## (nu functioneaza pentru ferestre 'inline')
            
            for angle in range(0, 360):
                ax.view_init(30, angle)
                plt.draw()
                plt.pause(0.001)
            
            plt.tight_layout()
            plt.show()
            
            
    print("\n\n- minime globale:")
    for minims in globalsMin:
        print("\tf(x, y) = {0}, (x, y) = ({1}, {2})".format(round(f(minims[0],minims[1]),3),round(minims[0],3),round(minims[1],3)))
    