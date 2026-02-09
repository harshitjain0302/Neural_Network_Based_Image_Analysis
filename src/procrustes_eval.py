import numpy as np
from scipy.spatial import procrustes
from scipy.stats import pearsonr

def load_mds_txt(path):
    """
    Loads human-provided ratings (expected: N x 8).
    Adjust parsing if your txt format differs.
    """
    arr = np.loadtxt(path)
    return arr

def procrustes_disparity(A, B):
    """
    Returns disparity after Procrustes alignment.
    Lower is better.
    """
    _, _, disparity = procrustes(A, B)
    return disparity

def dimwise_correlations(A, B):
    """
    Correlation per dimension after alignment.
    Assumes A and B already aligned shapes (N x D).
    """
    corrs = []
    for d in range(A.shape[1]):
        r, _ = pearsonr(A[:, d], B[:, d])
        corrs.append(r)
    return corrs
