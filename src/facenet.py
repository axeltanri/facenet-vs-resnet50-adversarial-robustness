import torch
from facenet_pytorch import InceptionResnetV1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

def get_distance(img1, img2):
    emb1 = model(img1)
    emb2 = model(img2)

    return torch.norm(emb1 - emb2).item()