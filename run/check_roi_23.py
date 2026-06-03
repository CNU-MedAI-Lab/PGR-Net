import os
import numpy as np
from glob import glob
from collections import defaultdict
from scipy.ndimage import label, find_objects
from tqdm import tqdm
import json

mask_paths = glob('/mnt/sdn/data/BraTS2023_new/train/Mask/*')

roi_clusters = []  # 每个元素是 List[[x1,y1,x2,y2], ...]
roi_counts = []  # 每张图的ROI数量统计

def cluster_center(cluster):
    xs = [(b[0] + b[2]) // 2 for b in cluster]
    ys = [(b[1] + b[3]) // 2 for b in cluster]
    return np.mean(xs), np.mean(ys)

def is_same_cluster(cluster, box, dist_thresh=40):
    cx1, cy1 = cluster_center(cluster)
    cx2 = (box[0] + box[2]) // 2
    cy2 = (box[1] + box[3]) // 2
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) < dist_thresh

for path in tqdm(mask_paths):
    mask = np.load(path)  # shape: [3, 160, 160]
    mask = np.any(mask, axis=0).astype(np.uint8)  # 合并三类为一个整体病灶区域

    labeled, num = label(mask)
    objects = find_objects(labeled)
    roi_counts.append(len(objects))

    for slc in objects:
        y1, y2 = slc[0].start, slc[0].stop
        x1, x2 = slc[1].start, slc[1].stop
        new_box = [x1, y1, x2, y2]

        matched = False
        for cluster in roi_clusters:
            if is_same_cluster(cluster, new_box):
                cluster.append(new_box)
                matched = True
                break
        if not matched:
            roi_clusters.append([new_box])

roi_candidates = []
for cluster in roi_clusters:
    centers = []
    max_size = 0
    for (x1, y1, x2, y2) in cluster:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        centers.append((cx, cy))
        max_size = max(max_size, max(x2 - x1, y2 - y1))
    avg_cx = int(np.mean([c[0] for c in centers]))
    avg_cy = int(np.mean([c[1] for c in centers]))
    roi_candidates.append({'center': [avg_cx, avg_cy], 'size': max_size})

with open("roi_candidates.json", "w") as f:
    json.dump(roi_candidates, f, indent=2)

print(f"共提取到 {len(roi_candidates)} 个候选病灶区域，已保存至 roi_candidates.json")
if roi_counts:
    print(f"图像中ROI数量 - 平均值: {np.mean(roi_counts):.2f}，最大值: {np.max(roi_counts)}")