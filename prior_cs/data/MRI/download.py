import cv2
import numpy as np
import os

# =====================================================
# 1. 视频路径
# =====================================================
video_path = "VIDEO/MRI.mp4"  # 改成你的视频路径
save_path = "video_patches.npy"  # 保存路径
patch_size = 32

cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "无法打开视频文件"

frames_patches = []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    H, W = gray.shape
    Ny = H // patch_size
    Nx = W // patch_size

    # 裁剪为整数个 patch
    gray_cropped = gray[:Ny*patch_size, :Nx*patch_size]

    # 划分 patch
    patches = []
    for y in range(Ny):
        row_patches = []
        for x in range(Nx):
            patch = gray_cropped[y*patch_size:(y+1)*patch_size,
                                 x*patch_size:(x+1)*patch_size]
            row_patches.append(patch)
        patches.append(row_patches)

    frames_patches.append(patches)
    frame_idx += 1

    if frame_idx % 10 == 0:
        print(f"Processed frame {frame_idx}")

cap.release()
print(f"Total frames processed: {frame_idx}")

# =====================================================
# 2. 转 numpy 并保存
# =====================================================
frames_array = np.array(frames_patches)  # shape: (num_frames, Ny, Nx, 32, 32)
np.save(save_path, frames_array)
print(f"Saved patches array to '{save_path}'")
print("Array shape:", frames_array.shape)
