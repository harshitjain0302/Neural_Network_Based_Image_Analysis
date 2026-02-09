from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

def run_kmeans(Z, k=3, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(Z)
    return km, labels

def run_gmm(Z, k=3, random_state=42):
    gmm = GaussianMixture(n_components=k, random_state=random_state)
    labels = gmm.fit_predict(Z)
    return gmm, labels

def cluster_metrics(Z, pred_labels, true_labels=None):
    out = {
        "silhouette": silhouette_score(Z, pred_labels),
        "davies_bouldin": davies_bouldin_score(Z, pred_labels),
    }
    if true_labels is not None:
        out["ari"] = adjusted_rand_score(true_labels, pred_labels)
    return out
