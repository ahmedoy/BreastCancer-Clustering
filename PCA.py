import numpy as np
from numpy import linalg as lg
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def MyPCA(X, k):
    # X is of shape (num_samples, num_features)
    X_mean = np.mean(X, axis=0).reshape(1, -1)
    X_norm = (X - X_mean)
    covariance = np.matmul(X_norm.transpose(), X_norm)
    eigenvalues, eigenvectors = lg.eig(covariance)
    idx = eigenvalues.argsort()[::-1]     
    sorted_vecs = (eigenvectors[:, idx])[:,:k]    
    eigenvalues = eigenvalues[idx]
    PCA_matrix = np.matmul(X_norm, sorted_vecs)    
    return (PCA_matrix, eigenvectors, eigenvalues)


def pca2(X, num_components):
    # Compute the covariance matrix
    cov_matrix = np.cov(X.T)
    
    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the top 'num_components' eigenvectors
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]
    
    # Project the data onto the selected eigenvectors
    projected_data = np.dot(X, selected_eigenvectors)
    
    return projected_data
    
    
    
    









X = np.random.randn(20, 10) 
X = (X - np.mean(X, axis=0).reshape(1, -1))

pca_mat, eigenvectors, eigenvalues = MyPCA(X, 2)
pca_mat3 = pca2(X, 2)


pca = PCA(n_components=2)
pca.fit(X)
pca_mat2 = pca.transform(X)



plt.scatter(pca_mat[:,0], pca_mat[:,1], c='r', marker='x')
plt.scatter(pca_mat3[:,0], pca_mat3[:,1], c='b')
plt.show()