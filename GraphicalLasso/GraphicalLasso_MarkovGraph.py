import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.collections
import matplotlib.pyplot as plt
import sklearn, sklearn.covariance,sklearn.cluster,sklearn.manifold

def cov2corr(cov):
    cov_nom = np.array(cov)
    D = np.diag(np.power(np.diag(np.array(cov_nom)),-0.5))
    return np.dot(np.dot(D,cov_nom),D)

def cov2partialcorr(cov):
    omega=np.linalg.inv(np.array(cov))
    D=np.diag(np.power(np.diag(omega),-0.5))
    partialcorr=-np.dot(np.dot(D,omega),D)
    #convert diagonal component　from -1 to 1
    partialcorr+=2*np.eye(cov.shape[0])
    return partialcorr

def GraphicalLasso(alpha=0.01,
                   mode='cd',
                   tol=0.0001,
                   enet_tol=0.0001,
                   max_iter=100, 
                   verbose=False, 
                   assume_centered=False):
    """Sparse inverse covariance estimation with an l1-penalized estimator.
    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.
    Parameters
    ----------
    alpha : positive float, default 0.01
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.
    mode : {'cd', 'lars'}, default 'cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.
    tol : positive float, default 1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped.
    enet_tol : positive float, optional
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'.
    max_iter : integer, default 100
        The maximum number of iterations.
    verbose : boolean, default False
        If verbose is True, the objective function and dual gap are
        plotted at each iteration.
    assume_centered : boolean, default False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.
    Attributes
    ----------
    covariance_ : array-like, shape (n_features, n_features)
        Estimated covariance matrix
    precision_ : array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.
    n_iter_ : int
        Number of iterations run.
    See Also
    --------
    graphical_lasso, GraphicalLassoCV"""
    model = sklearn.covariance.GraphicalLasso(alpha=alpha, 
                                              mode=mode, 
                                              tol=tol, 
                                              enet_tol=enet_tol, 
                                              max_iter=max_iter, 
                                              verbose=verbose, 
                                              assume_centered=assume_centered)
    return model

def MinMaxNormalization(X,axis=0):
    if axis is None:
        Xmax = X.values.max()
        Xmin = X.values.min()
        return (X-Xmin)/(Xmax - Xmin)
    elif axis==0:
        Xmax = X.max()
        Xmin = X.min() 
        return (df - Xmin) / (Xmax - Xmin)
    elif axis==1:
        Xmax = X.T.max()
        Xmin = X.T.min()
        return ((df.T - Xmin) /(Xmax - Xmin)).T
    else:
        print('IndexError: tuple index out of range')
        
def Standardization(X,axis=0,ddof=1):
    """
    Normalized by N-1 by default. This can be changed using the ddof argument. ddof: int,default 1
    """
    if axis is None:
        return (X - X.values.mean()) / X.values.std(ddof=ddof)
    elif axis==0:
        return (X - X.mean()) / X.std(ddof=ddof)
    elif axis==1:
        ((X.T - X.T.mean()) / X.T.std(ddof=ddof)).T
    else:
        print('IndexError: tuple index out of range')

def heatmap(val,**opt):
    """
    See also Seaborn==0.9.0
    """
    return sns.heatmap(val,**opt,)

class MarkovGraph:
    def __init__(self,X,model,vervose=True):
        self.names = X.columns.values
        self.X = X
        self.model = model
        self.non_zero_threshold = 0.02
        self.cluster = self.affinity_propagation(X=X,model=model,vervose=vervose)
        self.embedding,self.diag,self.partialcorr,self.non_zero_matrix,self.non_zero_threshold = self.LocallyLinearEmbedding(non_zero_threshold=self.non_zero_threshold)
    
    def affinity_propagation(self,X,model,vervose=True):
        _, self.labels = sklearn.cluster.affinity_propagation(model.covariance_)
        self.n_labels = self.labels.max()
        cluster = {}
        for i in range(self.n_labels+1):
            cluster['Cluster{0}'.format(i+1)] = self.names[self.labels==i]
            if vervose:
                print('Cluster {0}: {1}'.format(i+1,self.names[self.labels==i]))
        return cluster
    
    def LocallyLinearEmbedding(self,non_zero_threshold=0.02,random_state=0):
        node_position_model = sklearn.manifold.TSNE(n_components=2, init='pca',random_state=0)
        embedding = node_position_model.fit_transform(self.X.T).T
        d = 1./np.sqrt(np.diag(self.model.precision_))
        partialcorr = cov2partialcorr(self.model.covariance_)
        non_zero = (np.abs(np.triu(partialcorr,k=1))>non_zero_threshold)
        return embedding,d,partialcorr,non_zero,non_zero_threshold

    def set_LocallyLinearEmbeddingThreshold(self,non_zero_threshold,random_state=0):
        self.embedding,self.diag,self.partialcorr,self.non_zero_matrix,self.non_zero_threshold = self.LocallyLinearEmbedding(non_zero_threshold=non_zero_threshold,random_state=random_state)

    def set_node_position(self,pos):
        if set(pos.keys()) != set(self.names):
            print("List names do not match.")
            return -1
        val = pd.DataFrame(pos)[self.names]
        self.embedding = val.values

    def plot(self,facecolor='w',figsize=(10,8),marker_edgecolors='k',marker_cmap='Spectral',line_cmap='Spectral'):
        fig = plt.figure(1,facecolor=facecolor,figsize=figsize)
        fig.clf()
        ax = fig.add_axes([0,0,1,1])
        ax.axis('off')
        ax.scatter(self.embedding[0], self.embedding[1], s=100*self.diag**2,edgecolors=marker_edgecolors, c=self.labels,cmap=marker_cmap)
        start_idx, end_idx = np.where(self.non_zero_matrix)
        segments = [[self.embedding[:, start], self.embedding[:, stop]] for start, stop in zip(start_idx, end_idx)]
        values = np.abs(self.partialcorr[self.non_zero_matrix])
        lc = matplotlib.collections.LineCollection(segments,
                                                   zorder=0, cmap=line_cmap,
                                                   norm=plt.Normalize(0, .7 * values.max()))  
        lc.set_array(values)
        lc.set_linewidths(15 * values)
        ax.add_collection(lc)
        for index, (name, label, (x, y)) in enumerate(zip(self.names, self.labels, self.embedding.T)):
            dx = x - self.embedding[0]
            dx[index] = 1
            dy = y - self.embedding[1]
            dy[index] = 1
            this_dx = dx[np.argmin(np.abs(dy))]
            this_dy = dy[np.argmin(np.abs(dx))]
            if this_dx > 0:
                horizontalalignment = 'left'
                x = x + .002
            else:
                horizontalalignment = 'right'
                x = x - .002
            if this_dy > 0:
                verticalalignment = 'bottom'
                y = y + .002
            else:
                verticalalignment = 'top'
                y = y - .002
            ax.text(x, y, name, size=10,
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment,
                    bbox=dict(facecolor='w',
                              edgecolor=plt.cm.jet(label / float(self.n_labels)),
                              alpha=.6))
        ax.set_xlim(self.embedding[0].min() - .15 * self.embedding[0].ptp(),
                    self.embedding[0].max() + .10 * self.embedding[0].ptp())
        ax.set_ylim(self.embedding[1].min() - .03 * self.embedding[1].ptp(),
                    self.embedding[1].max() + .03 * self.embedding[1].ptp())
        return fig,ax
