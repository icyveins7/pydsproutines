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
            sel = np.argmax(self.scores[self.scoretypes[0]])
        elif self.scoretypes[0] == "db":
            sel = np.argmin(self.scores[self.scoretypes[0]])
        elif self.scoretypes[0] == "ch":
            raise NotImplementedError("Calinski Harabasz Score maximisation not available.")
                
        
            
    def cluster(self, x, guesses):
        if x.ndim == 1:
            x = x.reshape((-1,1)) # does not overwrite external argument array
        
        # We loop until all conditions are met
        while(True):
            # Cluster first
            
                
        

    


#%%
if __name__ == "__main__":
    print("Running test.")
    
    