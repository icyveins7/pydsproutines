#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:20:01 2022

@author: seolubuntu
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt


#%%
class ClusterEngine:
    def __init__(self, guesses, minClusterSize=None, minClusterFraction=None, scoretypes=["sil"]):
        self.guesses = guesses
        self.minClusterSize = minClusterSize
        self.minClusterFraction = minClusterFraction
        self.scoretypes = scoretypes
        self.scoretitles = {'sil': "Silhouette",
                            'ch': "Calinski Harabasz",
                            'db': "Davies Bouldin"}
        
    def _cluster1d(self, x):
  
        self.scores = {key: np.zeros(len(self.guesses)) for key in self.scoretypes}
        
        # Compute all requested scores
        for i in range(len(self.guesses)):
            model = KMeans(n_clusters=self.guesses[i]).fit(x)
            if 'sil' in self.scoretypes:
                self.scores['sil'][i] = metrics.silhouette_score(x, model.labels_, metric='euclidean')
            if 'ch' in self.scoretypes:
                self.scores['ch'][i] = metrics.calinski_harabasz_score(x, model.labels_)
            if 'db' in self.scoretypes:
                self.scores['db'][i] = metrics.davies_bouldin_score(x, model.labels_)
                
        # Only use the first scoretype to decide
        if self.scoretypes[0] == "sil":
            sel = np.argmax(self.scores[self.scoretypes[0]]) # This one is max
        elif self.scoretypes[0] == "db":
            sel = np.argmin(self.scores[self.scoretypes[0]]) # but this is min
        elif self.scoretypes[0] == "ch":
            raise NotImplementedError("Calinski Harabasz Score maximisation not available.") # will have to look for the kink, but seems unreliable
            
        return self.guesses[sel]
                
        
            
    def cluster(self, x: np.ndarray,
                ensure_sorted: bool=True,
                verbose: bool=False):
        
        
        if x.ndim == 1:
            x = x.reshape((-1,1)) # does not overwrite external argument array
        
        # We loop until all conditions are met
        idxUsed = np.arange(x.size) # Initialize using the entire array
        idxRemoved = []
        while(True):
            # Cluster first
            bestguess = self._cluster1d(x[idxUsed])
            # Make the model with best guess of no. of clusters
            bestmodel = KMeans(n_clusters=bestguess).fit(x[idxUsed])
            # Count the cluster sizes
            uniqueLabels = np.unique(bestmodel.labels_)
            numPerCluster = np.array([np.argwhere(bestmodel.labels_ == label).size for label in uniqueLabels])
            if verbose:
                print("\nFound %d clusters with members:" % uniqueLabels.size)
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
            
                    
            elif self.minClusterFraction is not None and np.min(numPerCluster)/np.max(numPerCluster) < self.minClusterFraction:
                # We again remove the smallest cluster
                clusterToRemove = np.argmin(numPerCluster)
                idxToRemove = np.argwhere(bestmodel.labels_ == clusterToRemove).flatten()
                
                if verbose:
                    print("Removed indices")
                    print(idxUsed[idxToRemove])
                    print("with corresponding values")
                    print(x[idxUsed[idxToRemove]])
                
                
            else:
                break
            
            # Remove those slated for removal
            # Append to removal list, using global indices
            idxRemoved.extend(idxUsed[idxToRemove])
            # Then delete it
            idxUsed = np.delete(idxUsed, idxToRemove)
            
        # Sort if desired
        if ensure_sorted:
            sortedIdx = np.argsort(bestmodel.cluster_centers_.flatten())
            bestmodel.labels_ = sortedIdx[bestmodel.labels_]
            bestmodel.cluster_centers_ = bestmodel.cluster_centers_[sortedIdx]
            
        return bestguess, bestmodel, np.array(idxRemoved), idxUsed
    
    @staticmethod
    def plotClusters(x, bestmodel, idxRemoved, idxUsed, t=None, colours=['r','b'],
                     ax=None, title="Clusters"):
        if t is None:
            t = np.arange(x.size) # just numbers them
            
        if ax is None: # Make a new plot with the title
            fig, ax = plt.subplots(1,1,num=title)
        else: # Else just plot on existing axes object
            fig = None
            
        # Plot all the points
        ax.plot(t, x, 'x')
        
        # Plot the outliers removed
        if idxRemoved.size > 0:
            ax.plot(t[idxRemoved], x[idxRemoved], 'ko', label='Outliers (%d)' % idxRemoved.size)
        
        # Plot the clusters via labels
        for i in range(bestmodel.n_clusters):
            si = idxUsed[np.argwhere(bestmodel.labels_ == i)]
            ax.plot(t[si], x[si], colours[i%len(colours)]+'s', markerfacecolor='none', label='Cluster %d (%d)' % (i, si.size))
            
        ax.legend()
        
        return fig, ax
    
    def plotScores(self, title:str="Cluster Scores"):
        '''
        Parameters
        ----------
        title : str, optional
            Figure title. The default is 'Cluster Scores'.

        Returns
        -------
        fig : pyplot figure
        
        ax : pyplot axes
        '''
        if title is None:
            title = "Cluster Scores"
        fig, ax = plt.subplots(len(self.scoretypes),1,num=title)
        if len(self.scoretypes) == 1:
            ax = [ax] # Hotfix for when only 1 score is used
        
        for i in range(len(self.scoretypes)):
            ax[i].plot(self.guesses, self.scores[self.scoretypes[i]])
            ax[i].set_title(self.scoretitles[self.scoretypes[i]])
            
        return fig, ax
                
                
        

    


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
    
    # Test the cluster engine
    cle = ClusterEngine(np.arange(2,10), minClusterSize=5, scoretypes=['sil', 'db'])
    bestguess, bestmodel, idxRemoved, idxUsed = cle.cluster(m, verbose=True)
    
    # View
    plt.close('all')
    cle.plotClusters(m, bestmodel, idxRemoved, idxUsed)
    
    # Plot the scores
    cle.plotScores()
    
    
    