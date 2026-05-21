import cv2
import torch

def preprocess(img_path, device):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (160, 160))
    img = img / 255.0

    img = torch.tensor(img).float().permute(2,0,1).unsqueeze(0)
    return img.to(device)