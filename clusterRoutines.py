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
        
    def _cluster(self, x):
  
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
        '''
        Parameters
        ----------
        x : np.ndarray
            Input array to be clustered.
        ensure_sorted : bool, optional
            Option to sort the cluster centers in ascending order. The default is True.
        verbose : bool, optional
            Verbose debug prints. The default is False.

        Returns
        -------
        bestguess : int
            Best fit number of clusters based on configured guesses (see init).
        bestmodel : KMeans
            The fitted model. Common use cases are to see the labels via bestmodel.labels_.
            See scikit-learn's documentation for more info.
        idxRemoved: np.ndarray
            The indices removed as outliers based on the configured options (see init).
        idxUsed : np.ndarray
            The remaining indices that are used in the 'bestmodel'.
        '''  
        
        if x.ndim == 1:
            x = x.reshape((-1,1)) # does not overwrite external argument array
        
        # We loop until all conditions are met
        
        idxUsed = np.arange(len(x)) # Initialize using the entire array
        idxRemoved = []
        while(True):
            # Cluster first
            bestguess = self._cluster(x[idxUsed])
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
                
                
#############
class ClusterEngine2D(ClusterEngine):
    def __init__(self, guesses, minClusterSize=None, minClusterFraction=None, scoretypes=["sil"]):
        super().__init__(guesses, minClusterSize, minClusterFraction, scoretypes)
        
    @staticmethod
    def convertComplexToInterleaved2D(x):
        if x.dtype == np.complex64:
            return x.view(np.float32).reshape((-1,2))
        elif x.dtype == np.complex128:
            return x.view(np.float64).reshape((-1,2))
        else:
            return x
        
    def cluster(self, x: np.ndarray,
                verbose: bool=False):
        
        # Check if input is complex, view as interleaved
        x = self.convertComplexToInterleaved2D(x)
            
        # Call the original cluster, but sorting is ignored for 2D
        bestguess, bestmodel, idxRemoved, idxUsed = super().cluster(x, False, verbose)
        
        return bestguess, bestmodel, idxRemoved, idxUsed
    
    @staticmethod
    def plotClusters(x, bestmodel, idxRemoved, idxUsed, colours=['r','b'],
                     ax=None, title="Clusters"):

        # Check if input is complex, view as interleaved
        x = ClusterEngine2D.convertComplexToInterleaved2D(x)

        if ax is None: # Make a new plot with the title
            fig, ax = plt.subplots(1,1,num=title)
        else: # Else just plot on existing axes object
            fig = None
            
        # Plot all the points
        ax.plot(x[:,0], x[:,1], 'x')
        
        # Plot the outliers removed
        if idxRemoved.size > 0:
            ax.plot(x[idxRemoved,0], x[idxRemoved,1], 'ko', label='Outliers (%d)' % idxRemoved.size)
        
        # Plot the clusters via labels
        for i in range(bestmodel.n_clusters):
            si = idxUsed[np.argwhere(bestmodel.labels_ == i)]
            ax.plot(x[si,0], x[si,1], colours[i%len(colours)]+'s', markerfacecolor='none', label='Cluster %d (%d)' % (i, si.size))
            
        ax.legend()
        
        return fig, ax
                
#############
class AngularClusterEngine(ClusterEngine2D):
    def __init__(self, guesses, minClusterSize=None, minClusterFraction=None, scoretypes=["sil"], minAngleSep=None):
        super().__init__(guesses, minClusterSize, minClusterFraction, scoretypes)
        self.minAngleSep = minAngleSep
        
    def cluster(self, x: np.ndarray,
                verbose: bool=False):
        
        # Same as 2D
        bestguess, bestmodel, idxRemoved, idxUsed = super().cluster(x,verbose)
        
        # Check extra conditionals
        if self.minAngleSep is not None:
            numCombined = 0
            for i in range(bestmodel.n_clusters):
                for j in range(i+1, bestmodel.n_clusters):
                    # Check if the current cluster is within angular range of the later clusters
                    angle_ij = np.arccos(
                        np.dot(
                            bestmodel.cluster_centers_[i],
                            bestmodel.cluster_centers_[j])
                        ) / np.linalg.norm(bestmodel.cluster_centers_[i]) / np.linalg.norm(bestmodel.cluster_centers_[j])
                    
                    if np.abs(angle_ij) < self.minAngleSep:
                        # We set all of cluster i to the later cluster j
                        bestmodel.labels_[bestmodel.labels_ == i] = j
                        # Decrement the number of clusters
                        numCombined += 1
                        
                        if verbose:
                            print("Cluster %d merged into %d" % (i,j))
                            
                        break
            
            # We correct the number of clusters
            bestmodel.n_clusters -= numCombined
            # Also shift the labels back to 0
            bestmodel.labels_ = bestmodel.labels_ - np.min(bestmodel.labels_)
            # And correct the best guess
            bestguess = bestmodel.n_clusters
            
            if verbose and numCombined > 0:
                print("Merges result finally in %d clusters instead" % bestguess)
        
        return bestguess, bestmodel, idxRemoved, idxUsed
    
    @staticmethod
    def plotClusters(x, bestmodel, idxRemoved, idxUsed, colours=['r','b'],
                     ax=None, title="Clusters"):
        
        fig, ax = ClusterEngine2D.plotClusters(x, bestmodel, idxRemoved, idxUsed, colours, ax, title)
        # Set a simple circle and axis limits
        theta = np.arange(0,2*np.pi,0.01)
        circle = np.vstack((np.cos(theta),np.sin(theta)))
        ax.plot(circle[0],circle[1],'k--')
        ax.axis([-1.1,1.1,-1.1,1.1])
        ax.set_aspect('equal')
        
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
    
    ## Angular Clustering
    # Generate some angles
    a0 = np.random.randn(100) * 0.01
    a1 = np.random.randn(10) * 0.01 + np.pi/2
    
    # Attach
    ma = np.hstack((a0,a1))
    ma_cplx = np.exp(1j*ma)
    
    # Test the cluster engine
    acle = AngularClusterEngine(np.arange(2,10), minAngleSep=np.pi/24)
    aguess, amodel, aRemoved, aUsed = acle.cluster(ma_cplx, verbose=True)
    acle.plotClusters(ma_cplx, amodel, aRemoved, aUsed, title="Angle Clusters")
    
    # Test the cluster engine when it's just one cluster
    aguess, amodel, aRemoved, aUsed = acle.cluster(ma_cplx[:a0.size], verbose=True)
    acle.plotClusters(ma_cplx[:a0.size], amodel, aRemoved, aUsed, title="Angle Clusters (Only 1)")
    
    
    
    
    
    
    
    