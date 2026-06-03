import os
import numpy as np
from glob import glob
from collections import defaultdict
from scipy.ndimage import label, find_objects
from tqdm import tqdm
import json

mask_paths = glob('/home/dew/user_datas/BraTS_19_all/MICCAI19_all/Mask/*')

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

nums_ntno = 0
for path in tqdm(mask_paths):
    npmask = np.load(path)  # shape: [3, 160, 160]
    WT_Label = npmask.copy()
    WT_Label[npmask == 1] = 1.
    WT_Label[npmask == 2] = 1.
    WT_Label[npmask == 4] = 1.
    TC_Label = npmask.copy()
    TC_Label[npmask == 1] = 1.
    TC_Label[npmask == 2] = 0.
    TC_Label[npmask == 4] = 1.
    ET_Label = npmask.copy()
    ET_Label[npmask == 1] = 0.
    ET_Label[npmask == 2] = 0.
    ET_Label[npmask == 4] = 1.
    # nplabel = np.empty((240, 240, 3))#之前切成160 现在临时改成240
    # nplabel = np.empty((160, 160, 3))
    nplabel = np.empty((npmask.shape[0], npmask.shape[1], 3))
    nplabel[:, :, 0] = WT_Label
    nplabel[:, :, 1] = TC_Label
    nplabel[:, :, 2] = ET_Label
    nplabel = nplabel.transpose((2, 0, 1))
    mask = nplabel
    # if mask == None:
    #     nums_ntno = nums_ntno + 1
    # print(nums_ntno)
    mask = np.any(mask, axis=0).astype(np.uint8)  # 合并三类为一个整体病灶区域

    labeled, num = label(mask)
    objects = find_objects(labeled)
    # print(objects)
    roi_counts.append(len(objects))

    for slc in objects:
        if slc is None:
            continue
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