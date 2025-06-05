import os
import torch
import h5py
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class H5Dataset(Dataset):
    """
    A Dataset that reads images and labels directly from
    train_HAM10000.h5 or test_HAM10000.h5.  Returns {'A': tensor, 'label': int}.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # No extra flags needed here
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt   = opt
        self.phase = opt.phase.lower()  # 'train' or 'test'

        # Decide which HDF5 file to open
        if self.phase == 'train':
            self.h5_path = os.path.join(opt.dataroot, 'train_HAM10000.h5')
        else:
            self.h5_path = os.path.join(opt.dataroot, 'test_HAM10000.h5')

        assert os.path.isfile(self.h5_path), f"HDF5 file not found: {self.h5_path}"
        self.h5_file = h5py.File(self.h5_path, 'r')

        # ─── 1) Load the 'images' dataset (possibly under a Group) ───
        group_or_dataset = self.h5_file['images']
        if isinstance(group_or_dataset, h5py.Group):
            # If 'images' is a group, pick its first sub‐dataset key
            if 'images' in group_or_dataset:
                ds = group_or_dataset['images']
            else:
                first_key = list(group_or_dataset.keys())[0]
                ds = group_or_dataset[first_key]
            self.images = ds
        else:
            # 'images' is already a single dataset
            self.images = group_or_dataset

        # ─── 2) Load the 'labels' group or dataset ───
        if 'labels' in self.h5_file:
            group_or_dataset = self.h5_file['labels']
            if isinstance(group_or_dataset, h5py.Group):
                # **Key change**: Keep the entire Group under self.labels,
                # not just one sub-dataset.
                self.labels = group_or_dataset
            else:
                # If it's a dataset (and not a scalar), use it directly
                if group_or_dataset.shape == ():
                    raise ValueError(f"'labels' in {self.h5_path} is a scalar, not an array")
                self.labels = group_or_dataset
        else:
            raise KeyError(f"'labels' not found in {self.h5_path}")

        # ─── 3) Build transformations: Resize→ToTensor→Normalize for RGB 32×32 ───
        transform_list = []
        # DCGAN was trained on 32×32 RGB images, so we must match that:
        transform_list += [transforms.Resize((opt.load_size, opt.load_size))]
        transform_list += [transforms.ToTensor()]  # [0–255]→[0–1]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # ─── 4) Read the raw image array; convert to PIL and force RGB ───
        arr = np.array(self.images[idx])  # shape might be (H,W) or (H,W,C)

        if arr.ndim == 2:
            # grayscale → force to RGB
            img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        elif arr.ndim == 3:
            # H×W×C: if C=1 or C=3, ensure RGB
            img = Image.fromarray(arr.astype(np.uint8))
            if img.mode != 'RGB':
                img = img.convert("RGB")
        else:
            raise RuntimeError(f"Unexpected image array shape: {arr.shape}")

        img_tensor = self.transform(img)  # now [3, 32, 32], normalized to [–1,1]

        item = {'A': img_tensor}

        # ─── 5) Fetch the label for this index ───
        if self.labels is not None:
            if isinstance(self.labels, h5py.Group):
                # Expect a sub‐dataset keyed by the string version of idx
                key = str(idx)
                if key in self.labels:
                    raw = self.labels[key][()]       # scalar or small array
                    label = int(np.array(raw))       # force to Python int
                else:
                    raise KeyError(f"Label key '{key}' not found in labels group.")
            else:
                # If labels was a flat dataset, index directly
                label = int(self.labels[idx])
            item['label'] = label

        return item
