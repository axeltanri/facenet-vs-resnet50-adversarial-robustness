from dataset_utils import (
    load_fixed_dataset,
    load_pairs
)

from preprocess_facenet import preprocess

from facenet import (
    model,
    get_distance,
    DEVICE
)

from roc_utils import (
    compute_distances,
    compute_roc,
    get_eer_threshold
)

from evaluation import (
    evaluate,
    calculate_far,
    calculate_frr,
    calculate_robust_accuracy
)

from attacks import fgsm_attack


DATASET_PATH = "dataset/lfw-fixed"
EPSILON = 0.03

data = load_fixed_dataset(DATASET_PATH)
print("Total identities:", len(data))


pairs, labels = load_pairs(
    "pairs/all_pairs.txt"
)
genuine_pairs, _ = load_pairs(
    "pairs/genuine_pairs.txt"
)
imposter_pairs, _ = load_pairs(
    "pairs/imposter_pairs.txt"
)

print("Total pairs:", len(pairs))
print("Total genuine pairs:", len(genuine_pairs))
print("Total imposter pairs:", len(imposter_pairs))


#ROC Analysis
distances, labels_np = compute_distances(
    pairs,
    labels,
    lambda x: preprocess(x, DEVICE),
    get_distance
)
fpr, tpr, thresholds, roc_auc = compute_roc(
    distances,
    labels_np
)
threshold, eer = get_eer_threshold(
    fpr,
    tpr,
    thresholds
)

print("ROC AUC:", roc_auc)
print("EER:", eer)
print("Threshold:", threshold)


baseline_acc = evaluate(
    pairs,
    labels,
    threshold,
    lambda x: preprocess(x, DEVICE),
    get_distance
)

print("Baseline Accuracy:", baseline_acc)


far = calculate_far(
    imposter_pairs,
    threshold,
    lambda x, d=DEVICE: preprocess(x, d),
    get_distance,
    fgsm_attack,
    model,
    EPSILON,
    DEVICE,
    "impersonation",
    use_clamp=True
)

print("FGSM FAR:", far)


frr = calculate_frr(
    genuine_pairs,
    threshold,
    lambda x, d=DEVICE: preprocess(x, d),
    get_distance,
    fgsm_attack,
    model,
    EPSILON,
    DEVICE,
    "dodging",
    use_clamp=True
)

print("FGSM FRR:", frr)

robust_acc = calculate_robust_accuracy(
    genuine_pairs,
    imposter_pairs,
    threshold,
    lambda x, d=DEVICE: preprocess(x, d),
    get_distance,
    fgsm_attack,
    model,
    EPSILON,
    DEVICE,
    use_clamp=True
)

print("FGSM Robust Accuracy:", robust_acc)