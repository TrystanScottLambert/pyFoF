"""Unsupervised evaluation metrics as per sklearn.metrics.cluster"""

# Original methods extended from sklearn.metrics.cluster, namely https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/cluster/_unsupervised.py
# Justification: Methods from sklearn.metrics were needed for the specific use-case of galaxy group detection and friends-of-friends algorithm evaluation, 
#                where certain fields such as the right ascension are cyclic in their coordinate representation, and a wrap mean method is needed, for example.
# Original Authors: Robert Layton <robertlayton@gmail.com>
#          Arnaud Fouchet <foucheta@gmail.com>
#          Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import functools
import numpy as np
import pandas as pd

from utils import redshift_catalog_mean, redshift_projected_unscaled_separation
#from fof import BaseFoF

from sklearn.utils import check_X_y, check_random_state, _safe_indexing
from sklearn.metrics import pairwise_distances_chunked, pairwise_distances
from sklearn.preprocessing import LabelEncoder

def check_number_of_labels(n_labels, n_samples):
    """Check that number of labels are valid. Because checking the number of labels is important.

    Parameters
    ----------
    n_labels : int
        Number of labels.
    n_samples : int
        Number of samples.
    """
    if not 1 < n_labels < n_samples:
        raise ValueError(
            "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)"
            % n_labels
        )


def redshift_silhouette_score(
    X, labels, *, metric="euclidean", sample_size=None, random_state=None, **kwds
):
    """Compute the mean Silhouette Coefficient of all samples.
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.  To clarify, ``b`` is the distance between a sample and the nearest
    cluster that the sample is not a part of.
    Note that Silhouette Coefficient is only defined if number of labels
    is ``2 <= n_labels <= n_samples - 1``.
    This function returns the mean Silhouette Coefficient over all samples.
    To obtain the values for each sample, use :func:`silhouette_samples`.
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.
    Read more in the :ref:`User Guide <silhouette_coefficient>`.
    Parameters
    ----------
    X : array-like of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`metrics.pairwise.pairwise_distances
        <sklearn.metrics.pairwise.pairwise_distances>`. If ``X`` is
        the distance array itself, use ``metric="precomputed"``.
    sample_size : int, default=None
        The size of the sample to use when computing the Silhouette Coefficient
        on a random subset of the data.
        If ``sample_size is None``, no sampling is used.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for selecting a subset of samples.
        Used when ``sample_size is not None``.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    Returns
    -------
    silhouette : float
        Mean Silhouette Coefficient for all samples.
    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_
    .. [2] `Wikipedia entry on the Silhouette Coefficient
           <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
    """
    if sample_size is not None:
        X, labels = check_X_y(X, labels, accept_sparse=["csc", "csr"])
        random_state = check_random_state(random_state)
        indices = random_state.permutation(X.shape[0])[:sample_size]
        if metric == "precomputed":
            X, labels = X[indices].T[indices].T, labels[indices]
        else:
            X, labels = X[indices], labels[indices]
    return np.mean(redshift_silhouette_samples(X, labels, metric=metric, **kwds))

def _redshift_silhouette_reduce(D_chunk, start, labels, label_freqs):
    """Accumulate silhouette statistics for vertical chunk of X.
    Parameters
    ----------
    D_chunk : array-like of shape (n_chunk_samples, n_samples)
        Precomputed distances for a chunk.
    start : int
        First index in the chunk.
    labels : array-like of shape (n_samples,)
        Corresponding cluster labels, encoded as {0, ..., n_clusters-1}.
    label_freqs : array-like
        Distribution of cluster labels in ``labels``.
    """
    # accumulate distances from each sample to each cluster
    clust_dists = np.zeros((len(D_chunk), len(label_freqs)), dtype=D_chunk.dtype)
    for i in range(len(D_chunk)):
        clust_dists[i] += np.bincount(
            labels, weights=D_chunk[i], minlength=len(label_freqs)
        )

    # intra_index selects intra-cluster distances within clust_dists
    intra_index = (np.arange(len(D_chunk)), labels[start : start + len(D_chunk)])
    # intra_clust_dists are averaged over cluster size outside this function
    intra_clust_dists = clust_dists[intra_index]
    # of the remaining distances we normalise and extract the minimum
    clust_dists[intra_index] = np.inf
    clust_dists /= label_freqs
    inter_clust_dists = clust_dists.min(axis=1)
    return intra_clust_dists, inter_clust_dists


def redshift_silhouette_samples(X, labels, *, metric="euclidean", **kwds):
    """Compute the Silhouette Coefficient for each sample.
    The Silhouette Coefficient is a measure of how well samples are clustered
    with samples that are similar to themselves. Clustering models with a high
    Silhouette Coefficient are said to be dense, where samples in the same
    cluster are similar to each other, and well separated, where samples in
    different clusters are not very similar to each other.
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.
    Note that Silhouette Coefficient is only defined if number of labels
    is 2 ``<= n_labels <= n_samples - 1``.
    This function returns the Silhouette Coefficient for each sample.
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters.
    Read more in the :ref:`User Guide <silhouette_coefficient>`.
    Parameters
    ----------
    X : array-like of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array.
    labels : array-like of shape (n_samples,)
        Label values for each sample.
    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a ``scipy.spatial.distance`` metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    Returns
    -------
    silhouette : array-like of shape (n_samples,)
        Silhouette Coefficients for each sample.
    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_
    .. [2] `Wikipedia entry on the Silhouette Coefficient
       <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
    """
    X, labels = check_X_y(X, labels, accept_sparse=["csc", "csr"])

    # Check for non-zero diagonal entries in precomputed distance matrix
    if metric == "precomputed":
        error_msg = ValueError(
            "The precomputed distance matrix contains non-zero "
            "elements on the diagonal. Use np.fill_diagonal(X, 0)."
        )
        if X.dtype.kind == "f":
            atol = np.finfo(X.dtype).eps * 100
            if np.any(np.abs(np.diagonal(X)) > atol):
                raise ValueError(error_msg)
        elif np.any(np.diagonal(X) != 0):  # integral dtype
            raise ValueError(error_msg)

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    check_number_of_labels(len(le.classes_), n_samples)

    kwds["metric"] = metric
    reduce_func = functools.partial(
        _redshift_silhouette_reduce, labels=labels, label_freqs=label_freqs
    )
    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func, **kwds))
    intra_clust_dists, inter_clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)

    denom = (label_freqs - 1).take(labels, mode="clip")
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom

    sil_samples = inter_clust_dists - intra_clust_dists
    with np.errstate(divide="ignore", invalid="ignore"):
        sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples)


def redshift_calinski_harabasz_score(X, labels, h0_value, metric='euclidean', column_names=None, wrap_columns=['ra'], **kwds):
    """Compute the Calinski and Harabasz score.
    It is also known as the Variance Ratio Criterion.
    The score is defined as ratio of the sum of between-cluster dispersion and
    of within-cluster dispersion.
    Read more in the :ref:`User Guide <calinski_harabasz_index>`.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    Returns
    -------
    score : float
        The resulting Calinski-Harabasz score.
    References
    ----------
    .. [1] `T. Calinski and J. Harabasz, 1974. "A dendrite method for cluster
       analysis". Communications in Statistics
       <https://www.tandfonline.com/doi/abs/10.1080/03610927408827101>`_
    """
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    check_number_of_labels(n_labels, n_samples)

    extra_disp, intra_disp = 0.0, 0.0
    mean = redshift_catalog_mean(X, column_names, wrap_columns)

    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = redshift_catalog_mean(cluster_k, column_names, wrap_columns)
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        intra_disp += np.sum((cluster_k - mean_k) ** 2)

    return (
        1.0
        if intra_disp == 0.0
        else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))
    )


def redshift_davies_bouldin_score(X, labels, h0_value, metric='euclidean', column_names=None, wrap_columns=['ra'], **kwds):
    """Compute the Davies-Bouldin score.
    The score is defined as the average similarity measure of each cluster with
    its most similar cluster, where similarity is the ratio of within-cluster
    distances to between-cluster distances. Thus, clusters which are farther
    apart and less dispersed will result in a better score.
    The minimum score is zero, with lower values indicating better clustering.
    Read more in the :ref:`User Guide <davies-bouldin_index>`.
    .. versionadded:: 0.20
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    Returns
    -------
    score: float
        The resulting Davies-Bouldin score.
    References
    ----------
    .. [1] Davies, David L.; Bouldin, Donald W. (1979).
       `"A Cluster Separation Measure"
       <https://ieeexplore.ieee.org/document/4766909>`__.
       IEEE Transactions on Pattern Analysis and Machine Intelligence.
       PAMI-1 (2): 224-227
    """

    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=float)
    for k in range(n_labels):
        cluster_k = _safe_indexing(X, labels == k)
        centroid = redshift_catalog_mean(cluster_k, column_names, wrap_columns)
        centroids[k] = centroid
        pairwise_distances_output = pairwise_distances(cluster_k, [centroid], metric=metric)
        pairwise_distances_output = pairwise_distances_output/h0_value
        intra_dists[k] = np.average(pairwise_distances_output)

    centroid_distances = pairwise_distances(centroids, metric=metric)
    centroid_distances = centroid_distances/h0_value

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)


def redshift_dunn_index(X, labels, h0_value, metric='euclidean', column_names=None, wrap_columns=['ra']):  
    # Method for calculating Dunn's index
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, n_dim = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    diam_lst = []  # List holding the diameter of the clusters
    dist_lst = []  # List that holds the smallest distance between points in different sets
    if len(dist_lst) == 0:
        min_dist = 1
    else:
        min_dist = min(dist_lst)

    centroids = np.zeros((n_labels, n_dim), dtype=float)

    for k in range(n_labels):
        cluster_k = _safe_indexing(X, labels == k)
        centroid = redshift_catalog_mean(cluster_k, column_names, wrap_columns)
        centroids[k] = centroid

    n_centroids = n_labels

    # Calculate the diameters of the clusters
    for k in range(n_centroids):
        clust_diam_lst = []  # List that holds the greatest distance between data in the set
        for i in range(n_samples):
            for j in range(n_samples):
                # If x and y points belong to the same cluster and cluster labels are equal to k
                if labels[i] == labels[j] and labels[i] == k:
                    #Add distance between x and y points to clust_diam_lst
                    clust_diam_lst.append(metric(X[i], X[j]))
                    diam_lst.append(max(clust_diam_lst))  # Add the largest element of clust_diam_lst to diam_lst
    max_diam = max(diam_lst)  # Assign the largest element of diam_lst to the variable max_diam

    # Calculate the smallest distance between points in different clusters
    for i in range(n_centroids):
        for j in range(i + 1, n_centroids):
            clust_dist_lst = []  # List that holds the distances of data between two sets
            for x in range(n_samples):
                for y in range(n_samples):
                    # Check if x is in cluster i and y is in cluster j. if so calculate the distance ij between the two clusters, later take the min
                    if labels[x] == i and labels[y] == j:
                        clust_dist_lst.append(metric(X[x], X[y]))
            dist_lst.append(min(clust_dist_lst))  # Add smallest element of clust_dist_lst to dist_lst

    dunn_index = min_dist / max_diam

    return dunn_index 
