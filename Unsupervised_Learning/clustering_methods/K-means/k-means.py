import numpy as np
import matplotlib.pyplot as plt
import sys
import math
# for coloring points and clusters
colorMap = plt.get_cmap('Accent')

# read points from file
inFile = open("points.txt", 'r')
number_of_observed_points = int(inFile.readline()) # will be 100 as the txt file has first line = 100 and is equal to total no# of entries
points = np.zeros([number_of_observed_points, 2])
for point_index in range(number_of_observed_points):
    pt = inFile.readline()
    pt = pt.split()
    points[point_index, 0] = float(pt[0]) # x
    points[point_index, 1] = float(pt[1]) # y

k_clusters = 3

# generate random colors, one for each cluster based on k_clusters values
clusterColor = colorMap(np.array(range(k_clusters))/k_clusters)

# initialize distances from each point to corresponding centroid
dist = np.zeros(number_of_observed_points)

# ids of clusters for each point: 
# clusterID[i] = -1 means point i doesn't belong to any cluster
# clusterID[i] = j means point i belongs to cluster j, j=[0..k_clusters-1]
clusterID = np.full(number_of_observed_points, -1)

# randomly assign points to clusters
clusterID = np.random.randint(0, k_clusters, number_of_observed_points)

centroids = np.zeros([k_clusters, 2]) # positions of centroids

# k_clustersmeans iterations

nrMaxIterations = 10

for iter in range(nrMaxIterations):

    ### compute new centroids
    avg = np.zeros([k_clusters, 2])
    clusters_vectors = np.zeros([k_clusters, 2])
    stop = False
    
    # centroids[j] = average of points i with clusterID[i] == j      
    for i in range(number_of_observed_points):
        avg[clusterID[i]] += points[i] # sum up the coordinates of random ID vector given to point[i]
        clusters_vectors[clusterID[i]] += 1
        
    for i in range(k_clusters):
        avg[i] = avg[i]/clusters_vectors[i]
        
        aux = abs(avg[i] - centroids[i])
        # break_clusters when centroid positions don't change significantly from previous values
        if aux[0]  > 0.1 and aux[1] > 0.1:
            stop = True
           
        
        centroids[i] = avg[i]
        
    if not stop:
        print("stop")
        break    
    
    ### assign points to clusters
    for i in range(number_of_observed_points):
        bestDistance = sys.maxsize
        for j in range(k_clusters):
            currentDistance = math.dist(points[i],centroids[j])
            
            if currentDistance < bestDistance:
                bestDistance = currentDistance
                clusterID[i] = j
         
     
    # for any point i, clusterID[i] = j, where j is the index of the centroid closest to i,  i = [0..N-1] , j =[0..k_clusters-1]

#end for

# plot points, centroids
pointColors = np.array(clusterColor[clusterID])
plt.scatter(points[:,0], points[:,1], color = pointColors, mark_clusterser = 'o', s = 10)
centroidColors = np.array(clusterColor[range(k_clusters)])
plt.scatter(centroids[:, 0], centroids[:, 1], color = centroidColors, mark_clusterser = 'x', s = 100)
plt.show()    







