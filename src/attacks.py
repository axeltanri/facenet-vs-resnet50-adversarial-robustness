import torch

def fgsm_attack(model,image1,image2,epsilon,attack_type="impersonation", use_clamp=True):

    # ENABLE GRADIENT
    image1.requires_grad = True

    # FORWARD PASS: runs both images thorugh the model to get embeddings and calculate Euclidean distances
    emb1 = model(image1)
    emb2 = model(image2)
    dist = torch.norm(emb1 - emb2)

    # ATTACK OBJECTIVE
    # Impersonation: minimize distance, make two different people look like the same person
    # Dodging: maximize distance, make a same person look like a different one 

    if attack_type == "impersonation":
        loss = -dist
    elif attack_type == "dodging":
        loss = dist
    else:
        raise ValueError("Invalid attack type")

    # BACKPROP: calculate the the gradient of loss with respect to every pixel
    model.zero_grad()       #clears any gradients from previous passes
    loss.backward()         #applies chain rule to compute loss changes when each pixel changes slightly


    # FGSM Attack
    grad = image1.grad.data.sign()          #find the smallest pixel changes that maximally trick the model
    adv_image = image1 + epsilon * grad

    # CLAMP
    if use_clamp:
        adv_image = torch.clamp(adv_image, 0, 1)        #to adjust pixel value back to 0 or 1 when they go out of valid range

    return adv_image.detach()