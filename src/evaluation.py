# EVALUATE ATTACK
def evaluate(pairs, labels, threshold, preprocess, get_distance):
    
    correct = 0

    for (p1, p2), label in zip(pairs, labels):
        img1 = preprocess(p1)
        img2 = preprocess(p2)

        dist = get_distance(img1, img2)
        pred = 1 if dist < threshold else 0

        if pred == label:
            correct += 1

    return correct / len(labels)

def calculate_far(imposter_pairs,threshold,preprocess,get_distance,attack_fn,model,epsilon,device,attack_type,use_clamp = True):

    false_accept = 0

    for img1_path, img2_path in imposter_pairs:

        img1 = preprocess(img1_path, device)
        img2 = preprocess(img2_path, device)

        adv_img1 = attack_fn(model, img1, img2, epsilon, attack_type, use_clamp)

        dist = get_distance(adv_img1, img2)

        # accepted as genuine
        if dist < threshold:
            false_accept += 1

    return false_accept / len(imposter_pairs)

def calculate_frr(genuine_pairs,threshold, preprocess,get_distance, attack_fn, model, epsilon, device, attack_type, use_clamp = True):

    false_reject = 0

    for img1_path, img2_path in genuine_pairs:

        img1 = preprocess(img1_path, device)
        img2 = preprocess(img2_path, device)

        adv_img1 = attack_fn(model, img1, img2, epsilon, attack_type, use_clamp)

        dist = get_distance(adv_img1, img2)

        # Rejected
        if dist > threshold:
            false_reject += 1

    return false_reject / len(genuine_pairs)

def calculate_robust_accuracy(genuine_pairs,imposter_pairs,threshold,preprocess,get_distance,attack_fn,model,epsilon,device,use_clamp = True):
    correct = 0
    total = 0

    for img1_path, img2_path in genuine_pairs:

        img1 = preprocess(img1_path, device)
        img2 = preprocess(img2_path, device)

        adv_img1 = attack_fn(model,img1,img2,epsilon,"dodging",use_clamp)

        dist = get_distance(adv_img1, img2)

        pred = 1 if dist < threshold else 0

        if pred == 1:
            correct += 1
        total += 1

    for img1_path, img2_path in imposter_pairs:

        img1 = preprocess(img1_path, device)
        img2 = preprocess(img2_path, device)

        adv_img1 = attack_fn(model,img1,img2,epsilon,"impersonation",use_clamp)

        dist = get_distance(adv_img1, img2)

        pred = 1 if dist < threshold else 0

        if pred == 0:
            correct += 1
        total += 1

    return correct / total

