import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.signal import find_peaks
from collections import Counter
from tqdm import tqdm

# BraTS mask路径（3通道：WT/TC/ET）
mask_paths = glob.glob('/mnt/sdn/data/BraTS2023_new/train/Mask/*.npy')

all_sizes = []

for mask_path in tqdm(mask_paths):
    mask = np.load(mask_path)  # shape: [3, H, W]

    # 对每个通道单独处理
    for i in range(mask.shape[0]):
        mask_slice = mask[i]
        labeled, num_features = label(mask_slice > 0)  # 连通区域标记

        for region_idx in range(1, num_features + 1):
            coords = np.argwhere(labeled == region_idx)
            if coords.size == 0:
                continue
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)

            width = xmax - xmin + 1
            height = ymax - ymin + 1
            square_size = max(width, height)  # 最大外接正方形
            if square_size < 20:
                continue
            all_sizes.append(square_size)

# 统计尺寸分布
size_counter = Counter(all_sizes)
sizes = np.array(sorted(size_counter.keys()))
counts = np.array([size_counter[s] for s in sizes])

# 寻找10个显著波峰
peaks, _ = find_peaks(counts, distance=5)  # 可调整distance避免相邻峰
peak_sizes = sizes[peaks]
peak_counts = counts[peaks]

# 取前10大峰
top10_indices = np.argsort(peak_counts)[-10:][::-1]
top10_peaks = peak_sizes[top10_indices]
top10_counts = peak_counts[top10_indices]

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(sizes, counts, marker='o', linestyle='-', color='b', label='Lesion Size Distribution')
plt.scatter(top10_peaks, top10_counts, color='r', s=50, label='Peaks')
for peak in top10_peaks:
    plt.axvline(x=peak, color='g', linestyle='--', linewidth=1)
plt.xlabel('Lesion Square Size (pixels)')
plt.ylabel('Count')
plt.title('Lesion Square Size Distribution (Filtered, Size >= 20)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('model/multi_mamba/counts.png', dpi=300)
plt.close()

img_size = 160  # 图像边长
print("Top 10 Peak Sizes (pixels), Counts, and Size/Image Ratio:")
for s, c in zip(top10_peaks, top10_counts):
    ratio = s / img_size
    print(f"Size: {s}, Count: {c}, Ratio: {ratio:.3f}")