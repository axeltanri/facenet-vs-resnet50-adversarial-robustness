Comparative Robustness Analysis of FaceNet and ResNet50-Based Face Embedding Models under FGSM Adversarial Attacks

This research project evaluates the robustness of face verification systems against adversarial attacks using two embedding models:

FaceNet
ResNet50-based face embedding model

The experiments analyze how FGSM adversarial perturbations affect verification performance in face verification systems.

Features
- Face verification using embedding distance
- ROC-AUC and EER evaluation
- FGSM adversarial attack implementation
- Impersonation and dodging attack scenarios
- FAR and FRR evaluation
- Robust accuracy analysis
- Multi-epsilon robustness testing
- Comparative analysis between FaceNet and ResNet50

Project Structure

project/
│
├── dataset/
│   └── lfw-fixed/
│
├── pairs/
│   ├── all_pairs.txt
│   ├── genuine_pairs.txt
│   └── imposter_pairs.txt
│
├── src/
│   ├── attacks.py
│   ├── dataset_utils.py
│   ├── evaluation.py
│   ├── roc_utils.py
│   │
│   ├── facenet.py
│   ├── preprocess_facenet.py
│   ├── main_facenet.py
│   │
│   ├── resnet.py
│   ├── preprocess_resnet.py
│   └── main_resnet.py
│
└── README.md

Hardware Requirements
- Processor : Intel Core i5 / AMD Ryzen 5 or equivalent
- RAM : minimum 8 GB
- Storage : minimum 10 GB free space
- GPU (optional) : NVIDIA GPU with CUDA support
- Operating System : Windows 10/11 or Linux

Software Requirements
FaceNet Environment
- Python 3.10+
- PyTorch
- facenet-pytorch
- OpenCV
- NumPy
- scikit-learn
- matplotlib

ResNet50 Environments
- Python 3.10+
- PyTorch
- timm
- OpenCV
- NumPy < 2.0
- scikit-learn
- matplotlib

Dataset Requirements
- Fixed identities
- Fixed genuine pairs
- Fixed imposter pairs

Experimental Condiguration
Facenet : 
- image size : 160 x 160
- Pixel normalization [0,1]
ResNet50 :
- Image size : 224 x 224
- ImageNet normalization

Evaluation Metrics:
- Attack Success Rate
- FAR
- FRR
- ROC-AUC
- EER
- Robust Accuracy

Adversarial Attack configuration
- epsilons = [0.001, 0.005, 0.01, 0.03]