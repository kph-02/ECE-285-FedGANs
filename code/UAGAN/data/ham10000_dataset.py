import os
import h5py
import torch
import numpy as np
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms


class Ham10000Dataset(BaseDataset):
    """Dataset class for the HAM10000 dataset"""
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options"""
        parser.add_argument('--ham10000_data_dir', type=str, default='./datasets/HAM10000_processed',
                           help='path to processed HAM10000 dataset')
        return parser

    def __init__(self, opt):
        """Initialize HAM10000 dataset class
        
        Parameters:
            opt (Option class) -- stores all the experiment flags
        """
        BaseDataset.__init__(self, opt)
        self.h5_path = os.path.join(opt.ham10000_data_dir, 
                                   'train_HAM10000.h5' if self.isTrain else 'test_HAM10000.h5')
        self.h5_file = h5py.File(self.h5_path, 'r')
        
        # Get available keys
        self.keys = list(self.h5_file['images'].keys())
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        
        Parameters:
            index - a random integer for data indexing
        
        Returns:
            a dictionary of data
        """
        key = self.keys[index]
        img_array = self.h5_file[f'images/{key}'][()]
        label = self.h5_file[f'labels/{key}'][()]
        
        # Convert to tensor and normalize
        img_tensor = self.transform(img_array)
        
        # Create data dictionary
        return {
            'A': img_tensor,  # Main image
            'B': img_tensor,  # Same image (for compatibility if needed)
            'A_paths': f"{self.h5_path}:{key}",
            'B_paths': f"{self.h5_path}:{key}",
            'label': label,
            'index': index,
        }

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.keys)


class Ham10000SplitDataset(BaseDataset):
    """Dataset class for HAM10000 split by acquisition site"""
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options"""
        parser.add_argument('--ham10000_data_dir', type=str, default='./datasets/HAM10000_processed',
                           help='path to processed HAM10000 dataset')
        parser.add_argument('--ham10000_site', type=str, default=None,
                           help='specific acquisition site to use (if None, uses combined dataset)')
        return parser

    def __init__(self, opt):
        """Initialize HAM10000 split dataset class"""
        BaseDataset.__init__(self, opt)
        
        # Determine file path based on site or combined
        if hasattr(opt, 'ham10000_site') and opt.ham10000_site is not None:
            self.h5_path = os.path.join(opt.ham10000_data_dir, f'train_HAM10000_site_{opt.ham10000_site}.h5')
        else:
            self.h5_path = os.path.join(opt.ham10000_data_dir, 
                                       'train_HAM10000.h5' if self.isTrain else 'test_HAM10000.h5')
        
        self.h5_file = h5py.File(self.h5_path, 'r')
        
        # Get available keys
        self.keys = list(self.h5_file['images'].keys())
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        key = self.keys[index]
        img_array = self.h5_file[f'images/{key}'][()]
        label = self.h5_file[f'labels/{key}'][()]
        
        # Convert to tensor and normalize
        img_tensor = self.transform(img_array)
        
        # Create data dictionary
        return {
            'A': img_tensor,
            'B': img_tensor,
            'A_paths': f"{self.h5_path}:{key}",
            'B_paths': f"{self.h5_path}:{key}",
            'label': label,
            'index': index,
        }

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.keys)