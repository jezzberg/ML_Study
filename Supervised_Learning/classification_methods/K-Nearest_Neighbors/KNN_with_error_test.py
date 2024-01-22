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

index =int(0.6 * len(points))

trainPoints = points[:index]
trainLabel = labels[:index]

testPoints = points[index:]
testLabel = labels[index:]

K = 3
KNN_weight = False # False - non weight , True - weight 


labelColors = ['red', 'green']
unlabeledColor = 'black'


pointColors = [labelColors[int(labels[i])] for i in range(N)]
plt.scatter(points[:,0], points[:,1], color = pointColors)



# determine class label of new point: 
wrongPoint = 0   
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
    print("\tKNN weight Error = "+str(err))
else:
    print("\tKNN NON-weight Error = "+str(err))
# err_non_weight /= len(test_points)
# err_weight /= len(test_points)

pointColorsMyPoints = [labelColors[int(testLabel[i])] for i in range(len(testPoints))]
plt.scatter(testPoints[:,0], testPoints[:,1], color = pointColorsMyPoints, s = 60, marker='x')
plt.show()