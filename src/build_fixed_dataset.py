import os
import shutil
import random

# This code is used to build fixed dataset. Run this when you have not generate fixed dataset.

random.seed(42)

SOURCE_DATASET = "dataset/lfw-deepfunneled"
TARGET_DATASET = "dataset/lfw-fixed"

MIN_IMAGES = 4
TOTAL_PEOPLE = 100

os.makedirs(TARGET_DATASET, exist_ok=True)

valid_people = []

# FIND VALID PEOPLE
for person in os.listdir(SOURCE_DATASET):
    person_path = os.path.join(SOURCE_DATASET, person)

    if not os.path.isdir(person_path):
        continue
    files = os.listdir(person_path)

    if len(files) >= MIN_IMAGES:
        valid_people.append(person)


selected_people = random.sample(valid_people, TOTAL_PEOPLE)


# COPY IMAGES
for person in selected_people:

    src_person_path = os.path.join(SOURCE_DATASET, person)
    dst_person_path = os.path.join(TARGET_DATASET, person)

    os.makedirs(dst_person_path, exist_ok=True)

    files = sorted(os.listdir(src_person_path))
    selected_files = files[:4]

    for file in selected_files:

        src_img = os.path.join(src_person_path, file)
        dst_img = os.path.join(dst_person_path, file)

        shutil.copy(src_img, dst_img)

print("\nDONE")