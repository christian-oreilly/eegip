import numpy as np
import pandas as pd
import pickle
import scipy
from scipy.cluster import hierarchy
from scipy.spatial import distance
from scipy.cluster.hierarchy import _LINKAGE_METHODS
from multiprocessing import Pool
from itertools import product, combinations
from functools import partial
from warnings import warn
from scipy.cluster.hierarchy import inconsistent
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

from .connectivity import get_map




def get_best_clustering_index(mat, linkage_method="ward", linkage=None, criterion="inconsistent", depth=2):
    if linkage is None:
        linkage = safe_linkage(mat, linkage_method)
    ts, clus_ind = get_clustering_profile(mat, linkage, criterion=criterion, depth=depth)
    return np.max(clus_ind)


"""
def get_clustering_index(mat, linkage=None, t=None, linkage_method="ward", labels=None, exclude_diag=True,
                         criterion="inconsistent", depth=2):

    if labels is None:
        if t is None:
            return get_best_clustering_index(mat, linkage_method=linkage_method, linkage=linkage,
                                             criterion=criterion, depth=depth)

        labels = scipy.cluster.hierarchy.fcluster(linkage, t, criterion=criterion, depth=depth)
    masks = []
    sum_squares = 0.0

    # within clusters sum of squares
    for cluster_label, count in zip(*np.unique(labels, return_counts=True)):
        if count == 1:
            continue
        mask = np.ones_like(mat).astype(bool)
        ind_cluster = np.where(labels == cluster_label)[0]
        i, j = list(zip(*[(i, j) for i, j in combinations(ind_cluster, 2)]))
        mask[i, j] = False
        mask[j, i] = False
        if not exclude_diag:
            mask[ind_cluster, ind_cluster] = False
        masks.append(mask)
        masked_mat = np.ma.masked_array(mat.values, mask)
        cluster_sum_squares = np.sum((masked_mat - np.mean(masked_mat))**2)

        if not np.ma.is_masked(cluster_sum_squares):
            sum_squares += cluster_sum_squares

    # outside clusters sum of squares
    if np.sum(np.prod(masks, axis=0)):
        mask = np.logical_not(np.prod(masks, axis=0))
        if exclude_diag:
            mask += np.diag([True] * mat.shape[0])
        masked_mat = np.ma.masked_array(mat.values, mask)
        rest_sum_squares = np.sum((masked_mat - np.mean(masked_mat))**2)
        if not np.ma.is_masked(rest_sum_squares):
            sum_squares += rest_sum_squares

    # maximal sum of square
    mask = np.zeros_like(mat).astype(bool)
    if exclude_diag:
        mask += np.diag([True] * mat.shape[0])
    masked_mat = np.ma.masked_array(mat.values, mask)
    max_ss = np.sum((masked_mat - np.mean(masked_mat))**2)

    return np.sqrt(1.0 - sum_squares/max_ss)

"""


def sum_of_squares(x):
    return np.sum((x - np.mean(x))**2)


def get_clustering_index(mat, linkage=None, t=None, linkage_method="ward", labels=None, exclude_diag=True,
                         criterion="inconsistent", depth=2, symmetric=True):

    if labels is None:
        if t is None:
            return get_best_clustering_index(mat, linkage_method=linkage_method, linkage=linkage,
                                             criterion=criterion, depth=depth)

        labels = scipy.cluster.hierarchy.fcluster(linkage, t, criterion=criterion, depth=depth)

    sum_squares = 0.0

    # within clusters sum of squares
    mask = np.zeros_like(mat, dtype=bool)
    for cluster_label, count in zip(*np.unique(labels, return_counts=True)):
        if count == 1:
            continue

        ind_cluster = np.where(labels == cluster_label)[0]
        i, j = list(zip(*list(combinations(ind_cluster, 2))))
        mask[j, i] = True
        sum_squares += sum_of_squares(mat.values[j, i])
        if not exclude_diag:
            mask[ind_cluster, ind_cluster] = True
            sum_squares += sum_of_squares(mat.values[ind_cluster, ind_cluster])
        if not symmetric:
            mask[i, j] = True
            sum_squares += sum_of_squares(mat.values[i, j])
        assert not np.isnan(sum_squares)

    # outside clusters sum of squares
    if exclude_diag:
        if symmetric:
            diag_sym_mask = np.triu(np.ones_like(mat, dtype=bool))
        else:
            diag_sym_mask = np.diag([True] * mat.shape[0], dtype=bool)
    else:
        if symmetric:
            diag_sym_mask = np.triu(np.ones_like(mat, dtype=bool), k=1)

    #if np.sum(mask):
    if exclude_diag or symmetric:
        mask += diag_sym_mask
    masked_mat = np.ma.masked_array(mat.values, mask)
    rest_sum_squares = sum_of_squares(masked_mat)
    if not np.ma.is_masked(rest_sum_squares):
        sum_squares += rest_sum_squares
    assert not np.isnan(sum_squares)

    # maximal sum of square
    mask = np.zeros_like(mat).astype(bool)
    if exclude_diag or symmetric:
        mask += diag_sym_mask
    masked_mat = np.ma.masked_array(mat.values, mask)
    max_ss = np.sum((masked_mat - np.mean(masked_mat))**2)

    # Without this check, rounding errors cause 1.0 - sum_squares/max_ss to become negative in some case and the
    # computed coefficient becomes NaN
    if np.abs(sum_squares - max_ss) < 1e-10:
        coef = 0.0
    else:
        coef = np.sqrt(1.0 - sum_squares/max_ss)

    #assert(coef < 1.0)
    assert(not np.isnan(coef))
    return coef


def get_clustering_profile(mat, linkage, start=0.0, stop=None, step=0.1, step_min=1e-13,
                           criterion="inconsistent", depth=2):

    if stop is None:
        # +steps because we want to have stop stictly larger than the the largest value in linkage[:, 2]
        if criterion == "inconsistent":
            stop = np.max(inconsistent(linkage, depth)[:, 3]) + step
        elif criterion == "distance":
            stop = np.max(cophenet(linkage, pdist(mat))[1]) + step
        else:
            stop = np.max(linkage[:, 2]) + step

    if step > (stop-start)/10.0:
        step = (np.max(linkage[:, 2])-start)/10.0

    # Small step_min start causing issues because of numerical rounding off issues
    if step_min < 1e-13:
        step_min = 1e-13

    change_points = get_clustering_profile_recur(linkage, start=start, stop=stop, step=step, step_min=step_min,
                                                 criterion=criterion, depth=depth)
    change_points.append(stop)

    clus_ind = [get_clustering_index(mat, linkage, t, criterion=criterion, depth=depth)
                for t in change_points]
    return change_points, clus_ind


def get_clustering_profile_recur(linkage, start, stop, step, step_min,
                                 start_labels=None, stop_labels=None, **f_cluster_kwargs):
    if step < step_min:
        return [start]

    diviser = 4.0

    # Inclusive of the stop point...
    ts = np.arange(start, stop+step, step)
    # ... but even when we include the stop point, we get some issues when the transition
    # is exactly at the stop point. Due probably to rounding off errors, sometime the transition is not found anymore
    # within the interval. To ensure we find it, we add a small value to the stop value.

    if start_labels is None:
        previous_labels = scipy.cluster.hierarchy.fcluster(linkage, ts[0], **f_cluster_kwargs)
    else:
        previous_labels = start_labels
    change_points = []
    for previous_t, t in zip(ts[:-1], ts[1:]):
        if t == ts[-1] and stop_labels is not None:
            labels = stop_labels
        else:
            labels = scipy.cluster.hierarchy.fcluster(linkage, t, **f_cluster_kwargs)
        if np.any(previous_labels != labels):
            change_points.extend(get_clustering_profile_recur(linkage, start=previous_t, stop=t, step=step/diviser,
                                                              step_min=step_min, start_labels=previous_labels,
                                                              stop_labels=labels, **f_cluster_kwargs))
            if len(np.unique(labels)) == 1:
                return change_points

        previous_labels = labels

    try:
        assert(len(change_points))
    except:
        print(start_labels)
        print(stop_labels)
        print(scipy.cluster.hierarchy.fcluster(linkage, start, **f_cluster_kwargs))
        print(scipy.cluster.hierarchy.fcluster(linkage, stop, **f_cluster_kwargs))
        print(scipy.cluster.hierarchy.fcluster(linkage, 0, **f_cluster_kwargs))

        from scipy.cluster.hierarchy import inconsistent
        depth = 2
        incons = inconsistent(linkage, depth)


        print(scipy.cluster.hierarchy.fcluster(linkage, np.max(incons[:, 3]), **f_cluster_kwargs))
        print(scipy.cluster.hierarchy.fcluster(linkage, 100, **f_cluster_kwargs))
        print(ts)
        raise

    return change_points


def attribute_labels(labels, clusters):

    counts = {}
    for no_cluster, cluster in enumerate(clusters):
        counts.update({(no_cluster, label): count
                       for label, count in zip(*np.unique(labels[cluster], return_counts=True))})
        counts[no_cluster, -no_cluster] = 0

    keys = product(*[np.concatenate([np.unique(labels[c]), [-no_c]]) for no_c, c in enumerate(clusters)])
    # Remove keys that are using the same labels for more than one cluster
    keys = [key for key in keys if len(np.unique(key)) == len(key)]
    scores = [np.sum([counts[no_cluster, label] for no_cluster, label in enumerate(key)]) for key in keys]
    ind_max_score = np.argmax(scores)
    return keys[ind_max_score], scores[ind_max_score]


def evaluate_clustering_methods(mat, real_clusters=None, generative_clusters=None, linkage_methods=None,
                                report_silhouette=True, report_best=True):

    if linkage_methods is None:
        linkage_methods = list(_LINKAGE_METHODS.keys())

    best_clus_ind = []
    best_ts = []
    best_nb_clusters = []
    accuracy = []
    accuracy_model = []
    ts_list_dict = {}
    clus_ind_dict = {}
    nb_clusters_dict = {}
    silhouette_scores = []

    for linkage_method in linkage_methods:
        linkage = safe_linkage(mat, linkage_method)
        ts, clus_ind = get_clustering_profile(mat, linkage, step_min=0.0000000001)
        labels = scipy.cluster.hierarchy.fcluster(linkage, t=ts[np.argmax(clus_ind)])
        nb_clusters = [len(labels_to_clusters(scipy.cluster.hierarchy.fcluster(linkage, t=t))) for t in ts]
        best_ts.append(ts[np.argmax(clus_ind)])
        best_nb_clusters.append(len(labels_to_clusters(labels)))
        best_clus_ind.append(np.max(clus_ind))
        if report_silhouette:
            if len(np.unique(labels)) == 1:
                silhouette_scores.append(np.nan)
            else:
                silhouette_scores.append(silhouette_score(mat, labels))

        if real_clusters is not None:
            nb_correct_class = attribute_labels(labels, real_clusters)[1]
            accuracy.append(np.sum(nb_correct_class) / len(np.concatenate(real_clusters)))

        if generative_clusters is not None:
            nb_correct_class = attribute_labels(labels, generative_clusters)[1]
            accuracy_model.append(np.sum(nb_correct_class) / len(np.concatenate(generative_clusters)))

        ts_list_dict[linkage_method] = ts
        clus_ind_dict[linkage_method] = clus_ind
        nb_clusters_dict[linkage_method] = nb_clusters

    # Adding a row for the best result
    if report_best:
        ind = np.argmax(best_clus_ind)
        linkage_methods.append("best")
        best_clus_ind.append(best_clus_ind[ind])
        best_nb_clusters.append(best_nb_clusters[ind])
        best_ts.append(np.nan)
        if real_clusters is not None:
            accuracy.append(accuracy[ind])
        if generative_clusters is not None:
            accuracy_model.append(accuracy_model[ind])
        if report_silhouette:
            silhouette_scores.append(silhouette_scores[ind])

    # Adding a row for the original data result (i.e., the clustering used for generating the matrix)
    if real_clusters is not None:
        linkage_methods.append("data")
        best_clus_ind.append(get_clustering_index(mat, labels=clusters_to_labels(real_clusters, mat.shape[0])))
        best_nb_clusters.append(len(real_clusters))
        best_ts.append(np.nan)
        accuracy.append(1.0)
        if generative_clusters is not None:
            labels = clusters_to_labels(real_clusters, mat.shape[0])
            nb_correct_class = attribute_labels(labels, generative_clusters)[1]
            accuracy_model.append(np.sum(nb_correct_class) / len(np.concatenate(generative_clusters)))
        if report_silhouette:
            silhouette_scores.append(silhouette_score(mat, clusters_to_labels(real_clusters, mat.shape[0])))

    # Adding a row for the original data result (i.e., the clustering used for generating the matrix)
    if generative_clusters is not None:
        linkage_methods.append("model")
        best_clus_ind.append(get_clustering_index(mat, labels=clusters_to_labels(generative_clusters, mat.shape[0])))
        best_nb_clusters.append(len(generative_clusters))
        best_ts.append(np.nan)
        accuracy_model.append(1.0)
        if real_clusters is not None:
            labels = clusters_to_labels(generative_clusters, mat.shape[0])
            nb_correct_class = attribute_labels(labels, real_clusters)[1]
            accuracy.append(np.sum(nb_correct_class) / len(np.concatenate(real_clusters)))
        if report_silhouette:
            silhouette_scores.append(silhouette_score(mat, clusters_to_labels(generative_clusters, mat.shape[0])))

    res_dict = {"recov_clus_ind": best_clus_ind, "recov_nb_clusters": best_nb_clusters,
                "link_method": linkage_methods, "optimal_t": best_ts}

    if real_clusters is not None:
        res_dict["accuracy"] = accuracy

    if generative_clusters is not None:
        res_dict["accuracy_model"] = accuracy_model

    if report_silhouette:
        res_dict["silhouette_scores"] = silhouette_scores

    results = pd.DataFrame(res_dict)

    return results, ts_list_dict, clus_ind_dict, nb_clusters_dict


def clusters_to_labels(clusters, nb_items):
    clusters = sorted(clusters, key=len, reverse=True)
    inds = np.arange(nb_items)
    inds_in_clusters = np.concatenate(clusters)
    labels = [[l] * len(c) for l, c in enumerate(clusters)]
    inds_not_in_clusters = inds[np.logical_not(np.in1d(inds, inds_in_clusters))]
    labels.append(np.arange(len(clusters), len(clusters) + len(inds_not_in_clusters)))

    ind_true_clustering = np.concatenate([inds_in_clusters, inds_not_in_clusters])
    labels = np.concatenate(labels)
    return labels[np.argsort(ind_true_clustering)]


def labels_to_clusters(labels):
    uniques, counts = np.unique(labels, return_counts=True)
    clusters = [np.where(labels == u)[0] for u in uniques]
    clusters = [c for c in clusters if len(c) > 1]
    return [c for c in clusters if len(c) > 1]


def generate_random_cluster_matrix(nb_item=30, scale=0.2, average_cluster_size=4):
    labels = np.arange(nb_item)
    np.random.shuffle(labels)
    cluster_lim = np.random.choice(np.arange(nb_item), int(nb_item / average_cluster_size), replace=False)
    cluster_lim = np.concatenate(([0], sorted(cluster_lim),  [nb_item]))
    #print(cluster_lim)
    #print([i, j for i, j in zip(cluster_lim[:-1], cluster_lim[1:])])
    clusters = [labels[i:j] for i, j in zip(cluster_lim[:-1], cluster_lim[1:])]
    # Clusters need to be made of at least two items
    clusters = [c for c in clusters if len(c) > 1]

    mat_to_cluster = np.random.normal(loc=0.0, scale=scale, size=(nb_item, nb_item))
    for cluster in clusters:
        for item1 in cluster:
            for item2 in cluster:
                # if item1 != item2:
                mat_to_cluster[item1, item2] = np.random.normal(loc=1.0, scale=scale)

    # Making the matrix symetric with zero diagonals. nans would be preferable on
    # the diagonal but currently, this makes crash the clusting
    # https://github.com/mwaskom/seaborn/issues/449 and, beside, the diagonal elements
    # are constraint to stay on the diagonal so they should not impact the clustering.
    mat_to_cluster = pd.DataFrame((mat_to_cluster + mat_to_cluster.T) / 2.0) - np.diag(np.diag(mat_to_cluster))

    return mat_to_cluster, clusters


def get_pij(clusters, nb_item, pd):
    labels = clusters_to_labels(clusters, nb_item)
    M = np.max(labels) + 1
    return (1.0 - pd) / (M - 1.0)

def get_new_flexible_cluster_randomization(clusters, nb_item, scale=0.2, pd_=1.0):

    rand_nums = np.random.rand(nb_item)
    labels = clusters_to_labels(clusters, nb_item)
    M = np.max(labels) + 1.0
    pij = (1.0 - pd_) / (M - 1.0)

    belonging_shift = np.zeros(nb_item, dtype=int)
    belonging_shift[rand_nums > pd_] = np.floor((rand_nums[rand_nums > pd_] - pd_) / pij).astype(int)

    new_labels = np.mod(labels + belonging_shift, M)
    new_clusters = labels_to_clusters(new_labels)

    return get_new_fixed_cluster_randomization(new_clusters, nb_item, scale), new_clusters


def get_new_fixed_cluster_randomization(clusters, nb_item, scale=0.2):

    mat_to_cluster = np.random.normal(loc=0.0, scale=scale, size=(nb_item, nb_item))
    for cluster in clusters:
        for item1 in cluster:
            for item2 in cluster:
                # if item1 != item2:
                mat_to_cluster[item1, item2] = np.random.normal(loc=1.0, scale=scale)

    # Making the matrix symetric with zero diagonals. nans would be preferable on
    # the diagonal but currently, this makes crash the clusting
    # https://github.com/mwaskom/seaborn/issues/449 and, beside, the diagonal elements
    # are constraint to stay on the diagonal so they should not impact the clustering.
    mat_to_cluster = pd.DataFrame((mat_to_cluster + mat_to_cluster.T) / 2.0) - np.diag(np.diag(mat_to_cluster))

    return mat_to_cluster


def identify_clusters(mat, linkage_method=None, linkage=None):
    if linkage is None:
        linkage = safe_linkage(mat, linkage_method)

    ts, clus_ind = get_clustering_profile(mat, linkage, step_min=0.0000000001)
    clusters = scipy.cluster.hierarchy.fcluster(linkage, t=ts[np.argmax(clus_ind)])

    labels = mat.index
    groups = []
    for unique in np.unique(clusters):
        groups.append(np.where(np.in1d(labels, mat.index[clusters == unique].tolist()))[0])
    return np.array(groups)


def safe_linkage(mat, linkage_method):
    if np.any(np.isnan(mat)):
        warn("NaN values detected in connectivity matrix. Zeroing these values.")
        mat[np.isnan(mat)] = 0.0

    if np.any(np.isinf(mat)):
        warn("Inf values detected in connectivity matrix. Zeroing these values.")
        mat[np.isinf(mat)] = 0.0

    return hierarchy.linkage(distance.pdist(mat), method=linkage_method)



def compute_one_exp_condition(args, sample_size, nb_iters, key_str):
    exp_key, con_mat = args
    if len(con_mat) == 0:
        # In some case con_mat is an empty list. This need to be investigated.
        warn("No matrix have been computed for {}.".format(exp_key))
        return []

    result_mats = []
    for n, nb_iter in zip(sample_size, nb_iters):
        for i in range(nb_iter):

            len(con_mat)
            if n == -1:
                mats = con_mat
            else:
                # For some reason choice on con_mat and indexing con_mat with list
                # is VERY SLOW. Maybe a bug
                # in xarray. Working much faster like this
                inds = np.random.choice(np.arange(len(con_mat)), n, replace=False)
                mats = [con_mat[i] for i in inds]

            try:
                mean_map, mask = get_map(mats, center=True)
            except ValueError:
                warn("The mats matrices contains complex for sample size {} of key {} contains imaginary values. "
                     .format(n, exp_key) + "Retaining only the absolute values.")
                mats = [np.absolute(mat.astype(str).astype(complex)) for mat in mats]
                mean_map, mask = get_map(mats, center=True)

            res = evaluate_clustering_methods(mean_map)[0]
            res["sample_size"] = n
            for key, val in zip(key_str, exp_key):
                res[key] = val
            result_mats.append(res)

    if len(result_mats) == 0:
        warn("compute_one_exp_condition failed to assess clustering for condition {}."
             .format(exp_key))
        return []

    return pd.concat(result_mats)


def compute_experiment_clustering_properties(file_name_in="con_matrices_for_clustering.pck",
                                             file_name_out="impact_of_averaging_on_clustering.pck",
                                             nb_processes=None, key_str=None, nb_iters=None, sample_size=None,
                                             small=False, **kwargs):

    if nb_processes is None:
        if small:
            nb_processes = 2
        else:
            nb_processes = 40
    if key_str is None:
        key_str = ["dataset", "event_type", "con_type", "fmin", "fmax", "method", "inv_method"]
    if nb_iters is None:
        nb_iters = [10, 5, 5, 5, 5, 5, 2, 2, 1, 1]
    if sample_size is None:
        sample_size = [1, 5, 10, 20, 30, 40, 50, 60, 75, -1]

    loop_experiments = partial(compute_one_exp_condition, sample_size=sample_size, nb_iters=nb_iters, key_str=key_str)

    with open(file_name_in, "rb") as f:
        con_matrices = pickle.load(f)

    if small:
        con_matrices = {key: val for key, val in list(con_matrices.items())[:2]}

    with Pool(processes=nb_processes) as pool:
        results = pd.concat([res_mat for res_mat in
                             pool.imap_unordered(loop_experiments, con_matrices.items())])
        results.to_pickle(file_name_out)
