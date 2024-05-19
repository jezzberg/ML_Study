import numpy as np
import matplotlib.pyplot as plt
import sys
import math
# for coloring points and clusters
colorMap = plt.get_cmap('Accent')

# read points from file
inFile = open('points.txt', 'r')
N = int(inFile.readline())
points = np.zeros([N, 2])
for i in range(N):
    pt = inFile.readline()
    pt = pt.split()
    points[i, 0] = float(pt[0]) #x
    points[i, 1] = float(pt[1]) #y

K = 3

# generate random colors, one for each cluster
clusterColor = colorMap(np.array(range(K))/K)

# initialize distances from each point to corresponding centroid
dist = np.zeros(N)

# ids of clusters for each point: 
# clusterID[i] = -1 means point i doesn't belong to any cluster
# clusterID[i] = j means point i belongs to cluster j, j=[0..K-1]
clusterID = np.full(N, -1)

#randomly assign points to clusters
clusterID = np.random.randint(0, K, N)

centroids = np.zeros([K, 2]) # positions of centroids

#kmeans iterations

nrMaxIterations = 10

for iter in range(nrMaxIterations):
    pass

    ### compute new centroids
    avg = np.zeros([K, 2])
    nrclusters = np.zeros([K, 2])
    stopSeq = False
    
    # centroids[j] = average of points i with clusterID[i] == j      
    for i in range(N):
        avg[clusterID[i]] += points[i]
        nrclusters[clusterID[i]] += 1
        
    for i in range(K):
        avg[i] = avg[i]/nrclusters[i]
        
        aux = abs(avg[i] - centroids[i])
        # break when centroid positions don't change significantly from previous values
        if aux[0]  > 0.1 and aux[1] > 0.1:
            stopSeq = True
           
        
        centroids[i] = avg[i]
        
    if not stopSeq:
        print("stop")
        break      
    
    ### assign points to clusters
    for i in range(N):
        bestDistance = sys.maxsize
        for j in range(K):
            currentDistance = math.dist(points[i],centroids[j])
            
            if currentDistance < bestDistance:
                bestDistance = currentDistance
                clusterID[i] = j
         
     
    # for any point i, clusterID[i] = j, where j is the index of the centroid closest to i,  i = [0..N-1] , j =[0..K-1]

#end for

# plot points, centroids
pointColors = np.array(clusterColor[clusterID])
plt.scatter(points[:,0], points[:,1], color = pointColors, marker = 'o', s = 10)
centroidColors = np.array(clusterColor[range(K)])
plt.scatter(centroids[:, 0], centroids[:, 1], color = centroidColors, marker = 'x', s = 100)
plt.show()    







