# PGR_Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation

This repository contains the official PyTorch implementation of **PGR-Net**, a prior-guided ROI reasoning framework for brain tumor MRI segmentation, accepted by the **CVPR 2026 main conference**.

PGR-Net is a prior-guided ROI reasoning network for brain tumor MRI segmentation, accepted by the CVPR 2026 main conference. The method is motivated by the severe spatial sparsity of tumor lesions in brain MRI, where large background regions often dominate feature learning and computation. To address this issue, PGR-Net introduces a data-driven spatial prior set, a hierarchical Top-K ROI decision mechanism, and the WinGS-ROI module to progressively localize high-confidence lesion regions and guide feature learning with ROI-aware spatial information. Built upon a windowed RetNet backbone, PGR-Net achieves accurate and efficient segmentation on BraTS 2019, BraTS 2023, and MSD Task01, while using only 8.64M parameters. 

# Paper link:

https://openaccess.thecvf.com/content/CVPR2026/html/Lu_PGR-Net_Prior-Guided_ROI_Reasoning_Network_for_Brain_Tumor_MRI_Segmentation_CVPR_2026_paper.html

The current main model is `UNetRetNet`, which takes four MRI modalities as input and predicts three tumor sub-regions:

- `WT`: Whole Tumor
- `TC`: Tumor Core
- `ET`: Enhancing Tumor

PGR-Net mainly includes data preprocessing, model training, model testing, prediction visualization, and metric evaluation scripts.

## 🔧 Environment

We recommend creating an independent Conda environment:

```bash
conda create -n pgr_net python=3.10 -y
conda activate pgr_net
pip install -r requirement.txt
```

## 📁 Repository Structure

```text
PGR_Net/
├── data_processing/              # BraTS raw NIfTI preprocessing scripts
│   ├── cut_MICCAI_for_ordered.py  # Convert BraTS2023 NIfTI data into .npy training data
│   └── cut_MICCAI_for_shuffled.py # Auxiliary shuffled 2D slicing script
├── model/                        # PGR_Net / UNetRetNet model definitions and ROI priors
│   ├── roi_candidates.json        # ROI prior candidates used by UNetRetNet
│   └── ...
├── run/                          # Training, testing, and dataset checking scripts
│   ├── train_region_pgr.py        # Training / fine-tuning script
│   ├── test_region_pgr.py         # Testing / evaluation script
│   └── dataset.py                 # Dataset loading logic
├── utils/                        # Losses, metrics, parameter statistics, and other tools
├── checkpoint/                   # Model weights, logs, and testing outputs
└── README.md
```

Commonly used scripts:

- `run/train_region_pgr.py`: train or continue training the model
- `run/test_region_pgr.py`: load model weights, evaluate on the validation set, and save metrics
- `data_processing/cut_MICCAI_for_ordered.py`: convert BraTS2023 NIfTI data into `.npy` files
- `run/dataset.py`: dataset reading and matching logic

## 📊 Dataset Preparation

The current training and testing scripts use a hard-coded dataset path:

```python
root_path = '/mnt/sdn/data/BraTS2023_new/'
```

Before running the code, make sure the dataset follows the structure below, or manually modify `root_path` in both `run/train_region_pgr.py` and `run/test_region_pgr.py`.

```text
BraTS2023_new/
├── train/
│   ├── Image/
│   └── Mask/
└── val/
    ├── Image/
    └── Mask/
```

The data files should be saved in `.npy` format. Image and mask filenames must correspond one-to-one.
`run/dataset.py` checks whether the image path and mask path end with the same filename.

Each sample should follow these shape requirements:

- Image: `[4, H, W]`, where the four channels correspond to the BraTS MRI modalities
- Mask: `[3, H, W]`, where the three channels correspond to `WT`, `TC`, and `ET`

The ROI prior in the current model is built for `160 x 160` images by default. Therefore, we recommend using the preprocessing script to generate `160 x 160` inputs.

## Data Preprocessing

If you use the raw BraTS2023 NIfTI data, please refer to:

```bash
python -m data_processing.cut_BraTS.
```

## 🚀 Training

Training can be performed using:

```bash
python -m run.train_region_pgr \
  --Model pgr \
  --epochs 300 \
  --batch-size 24 \
  --lr 3e-4 \
  --optimizer Adam
```

Before training, make sure the checkpoint directory exists:

```bash
mkdir -p checkpoint/pgr
```

Common arguments:

- `--Model`: checkpoint subdirectory name, default is `pgr`
- `--epochs`: number of training epochs, default is `300`
- `--early-stop`: early stopping patience, default is `50`
- `--batch-size`: batch size, default is `24`
- `--input-channels`: number of input channels, default is `4`
- `--lr`: initial learning rate, default is `3e-4`
- `--optimizer`: optimizer, supports `Adam` and `SGD`
- `--loss`: loss function name, default is `BCEDiceLoss`
- `--aug`: whether to enable data augmentation, default is `False`

Important:

The current `run/train_region_pgr.py` contains the following line:

```python
model.load_state_dict(torch.load(f'checkpoint/{args.Model}/modelbest.pth'))
```

Therefore, by default, the script expects an existing checkpoint:

```text
checkpoint/<Model>/modelbest.pth
```

This setting is more suitable for continuing training or fine-tuning.

If you want to train from scratch, please comment out or remove this line.

Training outputs are saved to:

```text
checkpoint/<Model>/
├── args.txt              # training arguments
├── args.pkl              # serialized arguments
├── log.csv               # training and validation logs
├── modelbest.pth         # best model according to validation dice_mean
└── <epoch>.pth           # snapshots saved every 50 epochs
```

## 🧪 Testing / Evaluation

To evaluate the best checkpoint:

```bash
python -m run.test_region_pgr \
  --Model pgr \
  --batch-size 16
```

To evaluate a specified checkpoint:

```bash
python -m run.test_region_pgr \
  --Model pgr \
  --checkpoint checkpoint/pgr/modelbest.pth \
  --batch-size 16
```

To save sample prediction visualizations:

```bash
python -m run.test_region_pgr \
  --Model pgr \
  --checkpoint checkpoint/pgr/modelbest.pth \
  --save-images \
  --save-mode samples \
  --save-examples 40
```

To save all prediction results:

```bash
python -m run.test_region_pgr \
  --Model pgr \
  --checkpoint checkpoint/pgr/modelbest.pth \
  --save-images \
  --save-mode all
```

Testing outputs are saved to:

```text
checkpoint/<Model>/
├── testlogall.csv
└── result/
    ├── input_samples/    # input image examples
    ├── label_samples/    # label examples
    ├── pred_samples/     # prediction examples
    ├── input/            # all input images, generated when --save-mode all
    ├── label/            # all label images, generated when --save-mode all
    └── pred/             # all prediction images, generated when --save-mode all
```

## 📈 Metrics

The testing script reports the following metrics for `WT`, `TC`, and `ET`:

- Dice Score
- Intersection over Union, IoU
- PPV / Precision
- Sensitivity / Recall
- Hausdorff Distance

The main training log fields include:

- `loss`: training loss
- `iou`: training IoU
- `val_loss`: validation loss
- `val_iou`: validation IoU
- `dice1`: WT Dice
- `dice2`: TC Dice
- `dice3`: ET Dice
- `dice_mean`: average Dice of the three tumor regions

## 🧠 Notes

- The current implementation is designed for BraTS-style brain tumor segmentation.
- The model takes four MRI modalities as input and predicts three tumor regions.
- The default ROI prior is built for `160 x 160` images.
- Please run scripts from the project root directory to avoid relative path errors.
- The current training script is configured for checkpoint-based continued training by default. For training from scratch, remove the pretrained checkpoint loading line.

## ❓ FAQ

### 1. Dataset not found

Please check whether `root_path` is correct and whether the following files exist:

```text
train/Image/*.npy
train/Mask/*.npy
val/Image/*.npy
val/Mask/*.npy
```

If not, modify `root_path` in:

```text
run/train_region_pgr.py
run/test_region_pgr.py
```

### 2. `modelbest.pth` not found during first training

The training script loads an existing checkpoint by default:

```python
model.load_state_dict(torch.load(f'checkpoint/{args.Model}/modelbest.pth'))
```

If you want to train from scratch, comment out or remove this line.

### 3. `model/roi_candidates.json` not found

`UNetRetNet` reads `model/roi_candidates.json` during initialization.

Please run the script from the project root directory:

```bash
python -m run.train_region_pgr
```

Do not run it directly inside the `run/` directory, otherwise relative paths may fail.


## 📄 Citation

If you find this project useful, please consider citing our work:

```bibtex
@inproceedings{Lu2026PGRNet,
  title     = {PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation},
  author    = {Lu, Jiacheng and Ding, Hui and Zhang, Shiyu and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
  pages     = {22816--22825}
}
```

Plain-text citation:

```text
Lu J, Ding H, Zhang S, et al. PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2026: 22816-22825.
```

## 📬 Contact

For questions or discussions, feel free to contact:

```text
Jiacheng Lu
Capital Normal University
Email: jchengl@foxmail.com
```
