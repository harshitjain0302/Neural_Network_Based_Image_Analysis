import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding as LLE, MDS
from sklearn.preprocessing import StandardScaler

def pca_embedding(X_flat, n_components=8, scale=True, random_state=42):
    X = StandardScaler().fit_transform(X_flat) if scale else X_flat
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(X)
    return Z, pca

def tsne_embedding(X_flat, n_components=2, perplexity=30, random_state=42):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    Z = tsne.fit_transform(X_flat)
    return Z, tsne

def lle_embedding(X_flat, n_components=2, n_neighbors=12, random_state=42):
    lle = LLE(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
    Z = lle.fit_transform(X_flat)
    return Z, lle

def mds_embedding(X_flat, n_components=2, random_state=42):
    mds = MDS(n_components=n_components, random_state=random_state)
    Z = mds.fit_transform(X_flat)
    return Z, mds
