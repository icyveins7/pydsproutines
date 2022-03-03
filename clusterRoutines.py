#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:20:01 2022

@author: seolubuntu
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics


#%%
class ClusterEngine:
    def __init__(self, minClusterSize=None, minClusterFraction=None, scoretypes=["sil"]):
        self.minClusterSize = minClusterSize
        self.minClusterFraction = minClusterFraction
        self.scoretypes = scoretypes
        
    def _cluster1d(self, x, guesses):
  
        self.scores = {key: np.zeros(len(guesses)) for key in self.scoretypes}
        
        for i in range(len(guesses)):
            model = KMeans(n_clusters=guesses[i]).fit(x)
            if 'sil' in self.scoretypes:
                self.scores['sil'][i] = metrics.silhouette_score(x, model.labels_, metric='euclidean')
            elif 'ch' in self.scoretypes:
                self.scores['ch'][i] = metrics.calinski_harabasz_score(x, model.labels_)
            elif 'db' in self.scoretypes:
                self.scores['db'][i] = metrics.davies_bouldin_score(x, model.labels_)
                
        # Use the first scoretype to decide
        if self.scoretypes[0] == "sil":
            sel = np.argmax(self.scores[self.scoretypes[0]]) # This one is max
        elif self.scoretypes[0] == "db":
            sel = np.argmin(self.scores[self.scoretypes[0]]) # but this is min
        elif self.scoretypes[0] == "ch":
            raise NotImplementedError("Calinski Harabasz Score maximisation not available.") # will have to look for the kink, but seems unreliable
            
        return guesses[sel]
                
        
            
    def cluster(self, x, guesses, verbose=False):
        if x.ndim == 1:
            x = x.reshape((-1,1)) # does not overwrite external argument array
        
        # We loop until all conditions are met
        idxUsed = np.arange(x.size) # Initialize using the entire array
        idxRemoved = []
        while(True):
            print("Clustering...")
            # Cluster first
            bestguess = self._cluster1d(x[idxUsed], guesses)
            # Make the model with best guess of no. of clusters
            bestmodel = KMeans(n_clusters=bestguess).fit(x[idxUsed])
            # Count the cluster sizes
            uniqueLabels = np.unique(bestmodel.labels_)
            numPerCluster = np.array([np.argwhere(bestmodel.labels_ == label).size for label in uniqueLabels])
            if verbose:
                print("Found %d clusters with members:" % uniqueLabels.size)
                print(numPerCluster)
            
            
            # Check for conditionals
            if self.minClusterSize is not None and np.any(numPerCluster < self.minClusterSize):
                # We remove the smallest cluster first
                clusterToRemove = np.argmin(numPerCluster)
                idxToRemove = np.argwhere(bestmodel.labels_ == clusterToRemove).flatten()
                
                if verbose:
                    print("Removed indices")
                    print(idxUsed[idxToRemove])
                    print("with corresponding values")
                    print(x[idxUsed[idxToRemove]])
                
                # Append to removal list, using global indices
                idxRemoved.extend(idxUsed[idxToRemove])
                # Then delete it
                idxUsed = np.delete(idxUsed, idxToRemove)
                    
            elif self.minClusterFraction is not None and np.min(numPerCluster)/np.max(numPerCluster) < self.minClusterFraction:
                # We again remove the smallest cluster
                clusterToRemove = np.argmin(numPerCluster)
                idxToRemove = np.argwhere(bestmodel.labels_ == clusterToRemove).flatten()
                
                if verbose:
                    print("Removed indices")
                    print(idxUsed[idxToRemove])
                    print("with corresponding values")
                    print(x[idxUsed[idxToRemove]])
                
                # Append to removal list, using global indices
                idxRemoved.extend(idxUsed[idxToRemove])
                # Then delete it
                idxUsed = np.delete(idxUsed, idxToRemove)
                
                
            else:
                break
            
        return bestguess, bestmodel, np.array(idxRemoved)
                
                
        

    


#%%
if __name__ == "__main__":
    print("Running test.")
    
    # Generate one feature
    f1 = np.random.randn(100)
    
    # Generate a little less of another feature
    f2 = np.random.randn(30) + 10
    
    # Generate some outliers
    o = np.array([-50,-90,-100])
    
    # Attach together
    m = np.hstack((f1,f2,o))
    np.random.shuffle(m) # shuffles in place
    
    # View
    import matplotlib.pyplot as plt
    plt.close('all')
    allIdx = np.arange(m.size)
    plt.plot(allIdx, m, 'x')
    
    # Test the cluster engine
    cle = ClusterEngine(minClusterSize=5)
    bestguess, bestmodel, idxRemoved = cle.cluster(m, np.arange(2,10), verbose=True)
    
    # Plot the outliers removed
    plt.plot(allIdx[idxRemoved], m[idxRemoved], 'ko')
    