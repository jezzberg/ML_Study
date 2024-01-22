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

# class labels are 0 - red , 1 - green

np.random.shuffle(dataset)
points = dataset[:, :2]
labels = dataset[:, 2]

K = 3
KNN_weight = False # False - non weight , True - weight 


labelColors = ['red', 'green']
unlabeledColor = 'black'


pointColors = [labelColors[int(labels[i])] for i in range(N)]
plt.scatter(points[:,0], points[:,1], color = pointColors)

#points to classify
myPoints = np.array([[6.2, 2.5], [5.37, 3.6], [4.65, 2.23]])
myLabels = np.zeros(len(myPoints))

# determine class label of new point: 
    
for i in range(len(myPoints)):
    distance = []
    for j in range(N):
        distance.append(math.dist(points[j], myPoints[i]))
    
    pointDist = zip(distance,labels)
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
        countGreen = 0
        for neighbor in neighbors:
            if neighbor[1] == 0:
                countRed += 1
            else:
                countGreen += 1
                
        if countGreen > countRed:
            myLabels[i] = 1
        else: 
            myLabels[i] = 0
        
    # with weight 
    else:
        wRed = 0
        wGreen = 0
        for neighbor in neighbors:
            if neighbor[1] == 0:
                wRed += 1/(neighbor[0] * neighbor[0])
            else:
                wGreen += 1/(neighbor[0] * neighbor[0])
                
        if wGreen > wRed:
            myLabels[i] = 1
        else: 
            myLabels[i] = 0



pointColorsMyPoints = [labelColors[int(myLabels[i])] for i in range(len(myPoints))]
plt.scatter(myPoints[:,0], myPoints[:,1], color = pointColorsMyPoints, s = 60, marker='x')
plt.show()