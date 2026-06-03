import numpy as np

import torch.utils.data


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        # self.imgfrq_paths = imgfrq_paths###################
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        # self.img_paths = sorted(self.img_paths)
        # self.mask_paths = sorted(self.mask_paths)
        self.aug = aug

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # imgfrq_path = self.imgfrq_paths[idx]
        img_path = self.img_paths[idx]
        # print(img_path)
        mask_path = self.mask_paths[idx]
        # print(img_path[-15:-4])
        if (img_path[-15:-4] != mask_path[-15:-4]):
            print("Get Data path error!")
        # print(mask_path)
        #读numpy数据(npy)的代码
        npimage = np.load(img_path).astype(np.float32)

        npmask = np.load(mask_path).astype(np.float32)
        # npimgfrq = np.load(img_path)

        # npimage[npimage == -9.0]=0.
        if np.max(npimage)==-9.0:
            npimage[npimage == -9.0]=0.
        else :
            npimage = (npimage - np.amin(npimage) + 0.00001) / (np.amax(npimage)-np.amin(npimage) + 0.00001)
        # print("np.min(npimage)",np.min(npimage))
    # fx = (fx - np.amin(fx) + 0.00001) / (np.amax(fx)-np.amin(fx) + 0.00001)

        return npimage,npmask
