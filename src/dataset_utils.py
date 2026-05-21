import os
import random
import itertools

random.seed(42)

def load_fixed_dataset(dataset_path):

    data = {}

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue
        images = [
            os.path.join(person_path, img)
            for img in os.listdir(person_path)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        data[person] = sorted(images)

    return data

def generate_balanced_pairs(data):

    genuine_pairs = []      #Two images of the same person, used in Dodging attack
    imposter_pairs = []     #Two images of different person, used in Impersonation

    people = list(data.keys())

    # GENUINE PAIRS
    for person in people:
        imgs = data[person]
        combinations = list(itertools.combinations(imgs, 2))
        for pair in combinations:
            genuine_pairs.append(pair)

    print("Total genuine pairs:", len(genuine_pairs))

    
    # IMPOSTER PAIRS
    while len(imposter_pairs) < len(genuine_pairs):
        person1, person2 = random.sample(people, 2)
        img1 = random.choice(data[person1])
        img2 = random.choice(data[person2])
        pair = (img1, img2)
        imposter_pairs.append(pair)

    print("Total imposter pairs:", len(imposter_pairs))

    # COMBINE
    pairs = genuine_pairs + imposter_pairs
    labels = [1] * len(genuine_pairs) + [0] * len(imposter_pairs)
    return pairs, labels, genuine_pairs, imposter_pairs

def load_pairs(file_path):
    
    pairs = []
    labels = []
    
    with open(file_path, "r") as f:
        for line in f:
            p1, p2, label = line.strip().split("|")
            pairs.append((p1, p2))
            labels.append(int(label))
    return pairs, labels

def save_pairs(pairs, labels, output_file):
    with open(output_file, "w") as f:
        for (img1, img2), label in zip(pairs, labels):
            line = f"{img1}|{img2}|{label}\n"
            f.write(line)
    print(f"Pairs saved to {output_file}")