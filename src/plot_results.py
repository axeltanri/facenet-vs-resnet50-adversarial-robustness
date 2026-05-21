import matplotlib.pyplot as plt

#Epsilons used in this experiment
epsilons = [0.001, 0.005, 0.01, 0.03]


# FaceNet
facenet_asr = [0.142, 0.407, 0.664, 0.921]
facenet_far = [0.162, 0.477, 0.735, 0.925]
facenet_frr = [0.123, 0.337, 0.593, 0.913]
facenet_robust_acc = [0.858, 0.593, 0.336, 0.079]


# ResNet50
resnet_asr = [0.432, 0.699, 0.782, 0.792]
resnet_far = [0.422, 0.630, 0.700, 0.667]
resnet_frr = [0.443, 0.768, 0.863, 0.918]
resnet_robust_acc = [0.568, 0.301, 0.218, 0.208]


# ASR GRAPH
plt.figure(figsize=(8,5))
plt.plot(
    epsilons,
    facenet_asr,
    marker='o',
    label='FaceNet'
)
plt.plot(
    epsilons,
    resnet_asr,
    marker='o',
    label='ResNet50'
)
plt.xlabel("Epsilon")
plt.ylabel("ASR")
plt.title("Attack Success Rate vs Epsilon")

plt.legend()
plt.grid(True)
plt.show()


# FAR GRAPH
plt.figure(figsize=(8,5))
plt.plot(
    epsilons,
    facenet_far,
    marker='o',
    label='FaceNet'
)
plt.plot(
    epsilons,
    resnet_far,
    marker='o',
    label='ResNet50'
)
plt.xlabel("Epsilon")
plt.ylabel("FAR")
plt.title("False Acceptance Rate vs Epsilon")

plt.legend()
plt.grid(True)
plt.show()

# FRR GRAPH
plt.figure(figsize=(8,5))

plt.plot(
    epsilons,
    facenet_frr,
    marker='o',
    label='FaceNet'
)

plt.plot(
    epsilons,
    resnet_frr,
    marker='o',
    label='ResNet50'
)

plt.xlabel("Epsilon")
plt.ylabel("FRR")
plt.title("False Rejection Rate vs Epsilon")

plt.legend()
plt.grid(True)
plt.show()


# ROBUST ACCURACY GRAPH
plt.figure(figsize=(8,5))

plt.plot(
    epsilons,
    facenet_robust_acc,
    marker='o',
    label='FaceNet'
)

plt.plot(
    epsilons,
    resnet_robust_acc,
    marker='o',
    label='ResNet50'
)

plt.xlabel("Epsilon")
plt.ylabel("Robust Accuracy")
plt.title("Robust Accuracy vs Epsilon")

plt.legend()
plt.grid(True)
plt.show()