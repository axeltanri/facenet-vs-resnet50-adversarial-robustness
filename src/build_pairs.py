from dataset_utils import (load_fixed_dataset, generate_balanced_pairs, save_pairs)

# CONFIG
DATASET_PATH = "dataset/lfw-fixed"
data = load_fixed_dataset(DATASET_PATH)

pairs, labels, genuine_pairs, imposter_pairs = (
    generate_balanced_pairs(data)
)

# SAVE
save_pairs(
    pairs,
    labels,
    "pairs/all_pairs.txt"
)

save_pairs(
    genuine_pairs,
    [1] * len(genuine_pairs),
    "pairs/genuine_pairs.txt"
)

save_pairs(
    imposter_pairs,
    [0] * len(imposter_pairs),
    "pairs/imposter_pairs.txt"
)

print("Fixed pairs generated")