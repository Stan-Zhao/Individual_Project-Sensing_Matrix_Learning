from torchvision import datasets, transforms
import numpy as np
import os

# ================= 配置 =================
DIGITS = [1, 3, 5, 7, 9]
MAX_PER_DIGIT = 5000   # 每个数字最多取多少个
SAVE_DIR = "./data/mniset_digits"

os.makedirs(SAVE_DIR, exist_ok=True)

# ================= 加载 MNIST =================
transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# ================= 初始化容器 =================
X_dict = {d: [] for d in DIGITS}

print("Collecting MNIST digits:", DIGITS)

# ================= 数据收集 =================
for img, label in dataset:
    if label in DIGITS:
        if len(X_dict[label]) < MAX_PER_DIGIT:
            x = img.view(-1).numpy()   # flatten -> (784,)
            X_dict[label].append(x)

    # 如果所有数字都收集够了，提前结束
    if all(len(X_dict[d]) >= MAX_PER_DIGIT for d in DIGITS):
        break

# ================= 保存 =================
for d in DIGITS:
    X = np.stack(X_dict[d], axis=1)  # (784, N)
    save_path = os.path.join(SAVE_DIR, f"X_digit_{d}.npy")
    np.save(save_path, X)
    print(f"Saved {save_path}, shape = {X.shape}")

print("All digit datasets saved successfully.")
