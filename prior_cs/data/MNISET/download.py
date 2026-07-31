from torchvision import datasets, transforms
import numpy as np

transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

num_images = 5000
X_list = []

for i in range(num_images):
    img, _ = dataset[i]
    x = img.view(-1).numpy()
    X_list.append(x)

X = np.stack(X_list, axis=1)   # shape = (784, 5)
import numpy as np

# 假设 X shape = (784, L)
np.save("X_train.npy", X)

print("Saved X_train.npy, shape:", X.shape)

