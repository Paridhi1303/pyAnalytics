# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 20:24:35 2021

@author: rukap
"""
#standard libaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

math=[20,50,25,35,40]
science=[25,45,22,40,35]
indexNo = ['S1','S2','S3','S4','S5']
df = pd.DataFrame({'math':math, 'science':science}, index=indexNo)
df
df.plot(kind='scatter', x='math', y='science')
plt.scatter(df['math'], df['science'], s = 20, c = 'k')
#s is used for size which may be an array or a scalar, c is for color with default value k which is black
help(plt.scatter)

from scipy.cluster.hierarchy import dendrogram , linkage
#Linkage Matrix
Z = linkage(df, method = 'ward')
Z

help(linkage)
#Perform hierarchical/agglomerative clustering using linkage. Format:linkage(y, method='single', metric='euclidean', optimal_ordering=False) The input y may be either a 1-D condensed distance matrix or a 2-D array of observation vectors

#plotting dendrogram
df
dendro = dendrogram(Z)
plt.title('Dendrogram')
plt.ylabel('Euclidean distance')
plt.show()
df

from scipy.spatial import distance
import numpy as np
#examples of euclidean distances
distance.euclidean([1, 0, 0], [0, 1, 0])
distance.euclidean([20,25],[25,22])  #closest : S1 with S2
np.sqrt(((20-25)**2 + (25-22)**2)) #sqrt(sum(x-y)^2)

distance.euclidean([20,25],[35,40]) 
distance.euclidean([20,25],[40,35])
distance.euclidean([35,40],[40,35])

#distance of all points in DF
from sklearn.neighbors import DistanceMetric
dist = DistanceMetric.get_metric('euclidean')
dist
df.to_numpy()
dist.pairwise(df.to_numpy())
#Kmeans clustering

df
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)
df
plt.scatter(df['math'], df['science'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
#c= kmeans.labels_.astype(float) gives different colors to different clusters
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, marker='D')
plt.show()
kmeans.fit(df)
kmeans = KMeans(n_clusters=3).fit(df)
kmeans.inertia_
kmeans.cluster_centers_  #average or rep values
kmeans.n_iter_  #in n times, clusters stabilised
kmeans.labels_
df

#iris dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram , linkage
 
#Getting the data ready
from pydataset import data
iris = data('iris')
df2 = iris.copy() 
df2.shape
df3 = df2.sample(5)
df3
df3.shape
df3.iloc[:,0:3].values
#Selecting certain features based on which clustering is done 
df4 = df3.iloc[:,0:3].values
df4

#Linkage Matrix
Z = linkage(df4, method = 'ward')

dist = DistanceMetric.get_metric('euclidean')
dist
dist.pairwise(df4)
 
kmeans = KMeans(n_clusters=2).fit(df4)
kmeans.inertia_
kmeans.cluster_centers_  #average or rep values
kmeans.n_iter_  #in n times, clusters stabilised
cn=kmeans.labels_
df4
df3.groupby(kmeans.labels_).mean()

#plotting dendrogram
dendro = dendrogram(Z)
plt.title('Dendrogram')
plt.ylabel('Euclidean distance')
plt.show()
df3

# cluster the mtcars data set into 2 groups, 3 groups
#find the average mileage, wt of these groups

from sklearn.cluster import AgglomerativeClustering
#agg_clustering = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
#predicting the labels
labels = agg_clustering.fit_predict(df)
labels