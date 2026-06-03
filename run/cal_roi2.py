import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import SimpleITK as sitk


img_path = '../../coms/image/sample_4331.png'

img_size = (160, 160)
# brain = np.load("BRATS_001.nii.gz_128.npy")
# brain = sitk.ReadImage("../../BraTS19_TCIA10_330_1_flair.nii.gz")
# brain = sitk.GetArrayFromImage(brain)
# brain = brain[80, :, :]
# brain = brain[brain.shape[0]//2 - 80:brain.shape[0]//2 + 80, brain.shape[1]//2 - 80:brain.shape[1]//2 + 80]

# 使用 PNG 图像作为切片输入
brain = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if brain is None:
    raise FileNotFoundError(f"无法加载图像文件: {img_path}")
brain = cv2.resize(brain, (160, 160))
brain = brain.astype(np.float32)
brain = (brain - brain.min()) / (brain.max() - brain.min() + 1e-8)


with open("roi_candidates_23.json", "r") as f:
    roi_candidates = json.load(f)

canvas = np.zeros((*img_size, 3), dtype=np.uint8)
# 使用蓝灰色调配色（从浅蓝到深灰蓝）
base_colors = [
    (100, 149, 237),  # cornflower blue
    (70, 130, 180),   # steel blue
    (60, 110, 160),   # medium blue-gray
    (50, 90, 140),    # darker blue-gray
    (40, 70, 120),    # deep gray-blue
]
def get_color(idx):
    c = base_colors[idx % len(base_colors)]
    return c

for idx, roi in enumerate(roi_candidates):
    cx, cy = roi["center"]
    size = roi["size"]
    half = size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img_size[1], cx + half)
    y2 = min(img_size[0], cy + half)
    color = get_color(idx)
    thickness = 2
    cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), color, thickness)

plt.figure(figsize=(6, 6))
plt.imshow(canvas)
plt.title("All ROIs Colored on Black Canvas")
plt.axis("off")
# plt.savefig('ROI_pic/ROIS.png', dpi=300)
plt.show()


plt.figure(figsize=(6, 6))
plt.imshow(brain, 'gray')
plt.title("All ROIs Colored on Black Canvas")
plt.axis("off")
# plt.savefig('ROI_pic/brain.png', dpi=300)
plt.show()


# Overlay ROI rectangles on the grayscale brain image
overlay = cv2.cvtColor((brain / brain.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
for idx, roi in enumerate(roi_candidates):
    cx, cy = roi["center"]
    size = roi["size"]
    half = size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img_size[1], cx + half)
    y2 = min(img_size[0], cy + half)
    color = get_color(idx)
    thickness = 2
    cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), color, thickness)

plt.figure(figsize=(6, 6))
plt.imshow(overlay)
plt.title("Gray Brain with Colored ROIs")
plt.axis("off")
# plt.savefig('ROI_pic/ROIS+brain.png', dpi=300)
plt.show()
