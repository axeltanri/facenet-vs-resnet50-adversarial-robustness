import torch
import torch.nn.functional as F

import timm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = timm.create_model(
    "resnet50",
    pretrained=True,
    num_classes=0
)

model.eval()
model.to(DEVICE)


def get_embedding(img):


    emb = model(img)

    emb = F.normalize(emb, p=2, dim=1)
    return emb


def get_distance(img1, img2):

    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)

    dist = torch.norm(emb1 - emb2, p=2)

    return dist.item()