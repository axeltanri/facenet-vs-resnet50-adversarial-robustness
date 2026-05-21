from PIL import Image

import torchvision.transforms as transforms

def preprocess_resnet(image_path, device):
    transform = transforms.Compose([

        transforms.Resize((224, 224)),
        transforms.ToTensor(),

        # ImageNet normalization
        # Required for pretrained ResNet
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])    

    image = Image.open(image_path).convert("RGB")
    image = transform(image)

    image = image.unsqueeze(0)

    return image.to(device)