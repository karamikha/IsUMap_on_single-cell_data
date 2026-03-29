import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import cross_val_score, StratifiedKFold


def pairwise_distance_correlation(X_high_dim: np.ndarray, X_low_dim: np.ndarray, metric_high: str = 'euclidean', metric_low: str = 'euclidean', method: str = 'spearman'):
    if X_high_dim.shape[0] != X_low_dim.shape[0]:
        raise ValueError('X_high_dim and X_low_dim must have the same number of samples')

    d_high = pdist(X_high_dim, metric=metric_high)
    d_low = pdist(X_low_dim, metric=metric_low)

    if method == 'spearman':
        corr, p_value = spearmanr(d_high, d_low)
    elif method == 'pearson':
        corr, p_value = pearsonr(d_high, d_low)
    else:
        raise ValueError('method must be spearman or pearson')

    return corr, p_value


def knn_accuracy(X_low_dim: np.ndarray, y: np.ndarray, n_neighbors: int = 10, cv: int = 5, random_state: int = 42):
    if X_low_dim.shape[0] != y.shape[0]:
        raise ValueError('X_low_dim and y must have the same number of samples')

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    scores = cross_val_score(clf, X_low_dim, y, cv=splitter, scoring='accuracy')
    return np.mean(scores), np.std(scores)


def knn_recall(X_high_dim: np.ndarray, X_low_dim: np.ndarray, n_neighbors: int = 10, metric: str = 'euclidean'):
    if X_high_dim.shape[0] != X_low_dim.shape[0]:
        raise ValueError('X_high_dim and X_low_dim must have the same number of samples')

    nn_high = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nn_high.fit(X_high_dim)
    high_idx = nn_high.kneighbors(X_high_dim, return_distance=False)[:, 1:]

    nn_low = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nn_low.fit(X_low_dim)
    low_idx = nn_low.kneighbors(X_low_dim, return_distance=False)[:, 1:]

    recalls = []
    for i in range(X_high_dim.shape[0]):
        recalls.append(len(set(high_idx[i]) & set(low_idx[i])) / n_neighbors)

    return np.mean(recalls)