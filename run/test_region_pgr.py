import os
import argparse
import warnings
import numpy as np
import torch

from glob import glob
from tqdm import tqdm
import pandas as pd
import cv2

from utils.metrics import dice_coef, iou_score, ppv, sensitivity
from run.dataset import Dataset
from model.pgr_net import UNetRetNet

try:
    from hausdorff import hausdorff_distance
except ImportError:
    hausdorff_distance = None
    print("[WARN] hausdorff_distance module not found, skipping Hausdorff metric")

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="BraTS2023", help='Dataset name')
    parser.add_argument('--Model', default="pgr", help='Model Name')
    parser.add_argument('--input-channels', default=4, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--checkpoint', default=None, help='Path to model checkpoint')
    parser.add_argument('--save-images', action='store_true', help='Save input, label and prediction images')
    parser.add_argument('--save-mode', default='samples', choices=['all', 'samples'], help='Image saving mode: all or samples')
    parser.add_argument('--save-examples', default=40, type=int, help='Number of sample images to save when --save-mode samples')
    parser.add_argument('--gpu-device', type=str, default='0')
    args = parser.parse_args()
    return args


def calculate_metrics(pb, gt):
    dice = dice_coef(pb, gt)
    iou = iou_score(pb, gt)
    precision = ppv(pb, gt)
    sens = sensitivity(pb, gt)
    haus = hausdorff_distance(gt, pb) if hausdorff_distance is not None else np.nan
    return dice, iou, precision, sens, haus


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    root_path = '/mnt/sdn/data/BraTS2023_new/'
    result_dir = f'checkpoint/{args.Model}/result'
    os.makedirs(result_dir, exist_ok=True)

    classification_records = []
    input_dir = os.path.join(result_dir, 'input')
    pred_dir = os.path.join(result_dir, 'pred')
    label_dir = os.path.join(result_dir, 'label')
    input_samples_dir = os.path.join(result_dir, 'input_samples')
    pred_samples_dir = os.path.join(result_dir, 'pred_samples')
    label_samples_dir = os.path.join(result_dir, 'label_samples')

    if args.save_images:
        for save_dir in [input_dir, pred_dir, label_dir, input_samples_dir, pred_samples_dir, label_samples_dir]:
            os.makedirs(save_dir, exist_ok=True)


    model = UNetRetNet(
        in_ch=args.input_channels,
        num_classes=3,
        base_ch=32,
        retention_heads=4,
        retention_dim=64,
        retention_at='bottleneck',  # 并行插入点（保留原有 RetNet）
        dropout=0.0,
        retention_window=8,
        retention_shift=0,  # 若想 Swin shift: 设为 8//2=4
        ret_replace_at=['enc4', 'bottleneck']  # 新增：把这些 block 的第2个卷积替换为 RetNet
    ).to(device)

    ckpt_path = args.checkpoint or f'checkpoint/{args.Model}/modelbest.pth'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"=> Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[WARN] Checkpoint not found at {ckpt_path}")
    model.eval()

    img_paths = sorted(glob(f'{root_path}/val/Image/*'))
    mask_paths = sorted(glob(f'{root_path}/val/Mask/*'))
    print(f"val_img_paths: {len(img_paths)} | val_mask_paths: {len(mask_paths)}")
    if args.save_images:
        print(f"=> Save images enabled: mode={args.save_mode}, samples={args.save_examples}")

    val_dataset = Dataset(args, img_paths, mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    metrics = {
        'wt_dice': [], 'tc_dice': [], 'et_dice': [],
        'wt_ppv': [], 'tc_ppv': [], 'et_ppv': [],
        'wt_sens': [], 'tc_sens': [], 'et_sens': [],
        'wt_iou': [], 'tc_iou': [], 'et_iou': [],
        'wt_haus': [], 'tc_haus': [], 'et_haus': []
    }

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.to(device)
            output = torch.sigmoid(model(input)).cpu().numpy()
            target = target.cpu().numpy()

            for j in range(output.shape[0]):
                pb, gt = output[j], target[j]
                pb[pb > 0.5] = 1
                pb[pb <= 0.5] = 0

                for k, name in enumerate(['wt', 'tc', 'et']):
                    d, iou, p, s, h = calculate_metrics(pb[k], gt[k])
                    metrics[f'{name}_dice'].append(d)
                    metrics[f'{name}_ppv'].append(p)
                    metrics[f'{name}_sens'].append(s)
                    metrics[f'{name}_iou'].append(iou)
                    metrics[f'{name}_haus'].append(h)

                if args.save_images:
                    global_idx = i * args.batch_size + j
                    save_name = f"{global_idx:06d}.png"
                    save_sample = global_idx < args.save_examples

                    if args.save_mode == 'all' or save_sample:
                        if args.save_mode == 'all':
                            cur_input_dir = input_dir
                            cur_label_dir = label_dir
                            cur_pred_dir = pred_dir
                        else:
                            cur_input_dir = input_samples_dir
                            cur_label_dir = label_samples_dir
                            cur_pred_dir = pred_samples_dir

                        # Save input image. Use the first input modality for visualization.
                        input_img = input[j, 0].detach().cpu().numpy()
                        input_img = input_img - input_img.min()
                        input_img = input_img / (input_img.max() + 1e-8)
                        input_img = (input_img * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(cur_input_dir, save_name), input_img)

                        # Save ground-truth segmentation as RGB mask: WT=R, TC=G, ET=B.
                        rgb_gt = np.zeros((gt.shape[1], gt.shape[2], 3), dtype=np.uint8)
                        rgb_gt[..., 0][gt[0] > 0.5] = 255
                        rgb_gt[..., 1][gt[1] > 0.5] = 255
                        rgb_gt[..., 2][gt[2] > 0.5] = 255
                        cv2.imwrite(os.path.join(cur_label_dir, save_name), rgb_gt)

                        # Save predicted segmentation as RGB mask: WT=R, TC=G, ET=B.
                        rgb_pred = np.zeros((pb.shape[1], pb.shape[2], 3), dtype=np.uint8)
                        rgb_pred[..., 0][pb[0] > 0.5] = 255
                        rgb_pred[..., 1][pb[1] > 0.5] = 255
                        rgb_pred[..., 2][pb[2] > 0.5] = 255
                        cv2.imwrite(os.path.join(cur_pred_dir, save_name), rgb_pred)
    mean_log = {k: np.nanmean(v) for k, v in metrics.items()}
    for name, value in mean_log.items():
        print(f"{name}: {value:.4f}")

    log_df = pd.DataFrame([mean_log])
    log_path = f'checkpoint/{args.Model}/testlogall.csv'
    log_df.to_csv(log_path, index=False)
    print(f"Saved log to {log_path}")


if __name__ == '__main__':
    main()

