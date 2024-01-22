import numpy as np
import matplotlib.pyplot as plt
import math

## read points from file
inFile = open('knnpoints.txt', 'r')
N = int(inFile.readline())
dataset = np.zeros([N, 3])
for i in range(N):
    pt = inFile.readline()
    pt = pt.split()
    dataset[i, 0] = float(pt[0]) #x
    dataset[i, 1] = float(pt[1]) #y
    dataset[i, 2] = float(pt[2]) #class label

#class labels are 0 - red , 1 - green

np.random.shuffle(dataset)
points = dataset[:, :2]
labels = dataset[:, 2]

index_1 =int(0.6 * len(points))
index_2 = int(0.2 * len(points))

trainPoints = points[:index_1]
trainLabel = labels[:index_1]

testPoints = points[index_1:(index_1+index_2)]
testLabel = labels[index_1:(index_1+index_2)]

validPoints = points[(index_1+index_2):]
validLabel = labels[(index_1+index_2):]

k_max = 10


KNN_weight = True # False - non weight , True - weight 


labelColors = ['red', 'green']
unlabeledColor = 'black'


pointColors = [labelColors[int(labels[i])] for i in range(N)]
plt.scatter(points[:,0], points[:,1], color = pointColors)



# determine class label of new point: 
wrongPoint = 0  
errCRT = {}

for K in range(1,k_max+1): 
    for i in range(len(validPoints)):
        distance = []
        for j in range(len(trainPoints)):
            distance.append(math.dist(trainPoints[j], validPoints[i]))
        
        pointDist = zip(distance,trainLabel)
        pointDist = list(pointDist)
        #print(set(pointDist))
        
        # distance - distance : color
        #sort by distance
        pointDist.sort(key = lambda x: x[0])
        #get k clossest points
        neighbors=pointDist[:K]
        
       
        
        if not KNN_weight:
        # non weight 
            countRed = 0
            countGreen =0
            for neighbor in neighbors:
                if neighbor[1] == 0:
                    countRed +=1
                else:
                    countGreen +=1
                    
            if countGreen > countRed and validLabel[i] != 1:
                wrongPoint += 1
            elif countGreen < countRed and validLabel[i] != 0:
                wrongPoint += 1
            
        # with weight 
        else:
            
            wRed = 0
            wGreen =0
            for neighbor in neighbors:
                try:
                    if neighbor[1] == 0:
                        wRed += 1/(neighbor[0]*neighbor[0])
                    else:
                        wGreen +=1/(neighbor[0]*neighbor[0])
                    
                except:
                    pass
            
            if wGreen > wRed and validLabel[i] != 1:
                wrongPoint += 1
            elif wGreen < wRed and validLabel[i] != 0: 
                wrongPoint += 1
    
    err = wrongPoint / len(validPoints)
    if KNN_weight: 
        print("\tVALIDATION: KNN weight Error = "+str(err)+" for k = "+str(K))
    else:
        print("\tVALIDATION: KNN NON-weight Error = "+str(err)+" for k = "+str(K))
    errCRT[K] = err
    wrongPoint = 0     

print(errCRT)

K = min(errCRT, key=errCRT.get)
print("Best K = "+str(K))

for i in range(len(testPoints)):
    distance = []
    for j in range(len(trainPoints)):
        distance.append(math.dist(trainPoints[j], testPoints[i]))
    
    pointDist = zip(distance,trainLabel)
    pointDist = list(pointDist)
    #print(set(pointDist))
    
    # distance - distance : color
    #sort by distance
    pointDist.sort(key = lambda x: x[0])
    #get k clossest points
    neighbors=pointDist[:K]
    
   
    
    if not KNN_weight:
    # non weight 
        countRed = 0
        countGreen =0
        for neighbor in neighbors:
            if neighbor[1] == 0:
                countRed +=1
            else:
                countGreen +=1
                
        if countGreen > countRed and testLabel[i] != 1:
            wrongPoint += 1
        elif countGreen < countRed and testLabel[i] != 0:
            wrongPoint += 1
        
    # with weight 
    else:
        
        wRed = 0
        wGreen =0
        for neighbor in neighbors:
            try:
                if neighbor[1] == 0:
                    wRed += 1/(neighbor[0]*neighbor[0])
                else:
                    wGreen +=1/(neighbor[0]*neighbor[0])
                
            except:
                pass
        
        if wGreen > wRed and testLabel[i] != 1:
            wrongPoint += 1
        elif wGreen < wRed and testLabel[i] != 0: 
            wrongPoint += 1

err = wrongPoint / len(testPoints)
if KNN_weight: 
    print("\tTEST: KNN weight Error = "+str(err))
else:
    print("\tTEST: KNN NON-weight Error = "+str(err))
# err_non_weight /= len(test_points)
# err_weight /= len(test_points)


pointColorsMyPoints = [labelColors[int(testLabel[i])] for i in range(len(testPoints))]
plt.scatter(testPoints[:,0], testPoints[:,1], color = pointColorsMyPoints, s = 60, marker='x')
plt.show()