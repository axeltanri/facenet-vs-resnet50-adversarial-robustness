import numpy as np
from sklearn.metrics import roc_curve, auc

def compute_distances(pairs, labels, preprocess, get_distance):
    distances = []

    for (p1, p2), _ in zip(pairs, labels):
        img1 = preprocess(p1)
        img2 = preprocess(p2)

        dist = get_distance(img1, img2)
        distances.append(dist)

    return np.array(distances), np.array(labels)


def compute_roc(distances, labels):
    scores = -distances
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, thresholds, roc_auc


def get_eer_threshold(fpr, tpr, thresholds):
    fnr = 1 - tpr
    idx = np.nanargmin(abs(fpr - fnr))

    eer = fpr[idx]
    threshold = -thresholds[idx]

    return threshold, eer