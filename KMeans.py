import numpy as np
import time  
import matplotlib.pyplot as plt  
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance
import random
from numpy.linalg import norm

# calculate EMD  
def euclDistance(vector1, vector2):  
	return sqrt(sum(power(vector2 - vector1, 2)))

def EMDistance(vector1, vector2): 
	d = wasserstein_distance(vector1, vector2)
	return d
	
def cosdistance(vector1, vector2):
    return 1-np.dot(vector1, vector2)/(norm(vector1)*norm(vector2))


# init centroids with random samples  
def initCentroids(dataSet, k):  
	print('init centroids with %d different random samples...' % k)
	numSamples, dim = dataSet.shape 
	centroids = np.zeros((k, dim))
	temp, uniq_cnt = np.unique(dataSet,axis=0, return_counts=True)
	temp = temp[uniq_cnt==1]
	for i in range(k): 
	    index = int(np.random.uniform(0, len(temp))) 
	    centroids[i, :] = temp[index, :] 
	return centroids  
  
# k-means cluster 
def kmeans(dataSet, k):  
	numSamples = dataSet.shape[0]  
    # first column stores which cluster this sample belongs to,  
    # second column stores the error between this sample and its centroid  
	clusterAssment = np.mat(np.zeros((numSamples, 2)))  
	clusterChanged = True  
  
    ## step 1: init k different centroids  
	centroids = initCentroids(dataSet, k)  
# 	print(centroids,'centroids')
	it = 0
	totalit = 1000
	while clusterChanged and it<totalit:  
		it += 1
		clusterChanged = False  
        ## for each sample  
		for i in range(numSamples):  #range
			minDist  = 1000000.0  
			minIndex = 0  
            ## for each centroid  
            ## step 2: find the centroid who is closest
			for j in range(k):  
				distance = cosdistance(centroids[j, :], dataSet[i, :])  
				if distance < minDist:  
					minDist  = distance  
					minIndex = j
				
            ## step 3: update its cluster 
			if clusterAssment[i, 0] != minIndex:  
				clusterChanged = True  
				clusterAssment[i, :] = minIndex, minDist #**2
# 		print('clusterAssment',it, clusterAssment, centroids)
  
        ## step 4: update centroids  
		for j in range(k):  
			index = np.array(clusterAssment[:, 0] == j).reshape(-1)
			pointsInCluster = dataSet[index] 
# 			print('pointsInCluster ', j, pointsInCluster)
			# centroids: the cloest point to mean or just cluster mean
			clusterMean = np.mean(pointsInCluster, axis = 0) 
# 			centroids[j, :] = clusterMean
			dists = np.zeros((pointsInCluster.shape[0]))
			for kk in range(pointsInCluster.shape[0]):
				dists[kk] = EMDistance(clusterMean, pointsInCluster[kk, :]) 
# 			dists = np.sum((pointsInCluster - clusterMean)**2, axis=1)
			centroids[j, :] = pointsInCluster[np.argmin(dists)] #np.mean(pointsInCluster, axis = 0)  
	print ('Congratulations, cluster complete!')  
	return centroids, clusterAssment  

# show your cluster only available with 2-D data 
def showCluster(dataSet, centroids, clusterAssment, name):  
	numSamples, dim = dataSet.shape  
	pca = PCA(n_components=2)
	pca.fit(dataSet)
	dataSet = pca.transform(dataSet)
# 	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr'] 
	mark = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan'] 

	# draw all samples  
	plt.figure()
	print('# cluster', len(set(clusterAssment)))
	for i in range(numSamples):
		markIndex = clusterAssment[i]
		if markIndex is None:
		    plt.plot(dataSet[i, 0], dataSet[i, 1], marker='o', color='black') 
		else:
		    plt.plot(dataSet[i, 0], dataSet[i, 1], marker='o', color=mark[int(markIndex)%len(mark)]) 

	# mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
# 	for i in range(k):  
# 		plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 10)  
	plt.savefig('./data/pressure_3D_24_24_1200_v2/%s.png' % name)
	plt.close()

