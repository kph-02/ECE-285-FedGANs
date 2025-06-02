import os
import torch
import h5py
from data.base_dataset import BaseDataset
from torchvision import transforms

class Ham10000MultisiteDataset(BaseDataset):
    """Dataset class for HAM10000 split by acquisition sites."""
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=32, crop_size=32, n_class=7)
        return parser
        
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot)
        self.num_sites = 10
        self.h5_files = []
        self.keys_per_site = []
        
        # Standard normalization for skin lesion images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load all site files
        for i in range(self.num_sites):
            h5_path = os.path.join(self.dir_AB, f'train_HAM10000_site_{i}.h5')
            if os.path.exists(h5_path):
                h5_file = h5py.File(h5_path, 'r')
                self.h5_files.append(h5_file)
                keys = list(h5_file['images'].keys())
                self.keys_per_site.append(keys)
                print(f"Site {i}: loaded {len(keys)} samples")
            else:
                print(f"Warning: File not found {h5_path}")
                self.h5_files.append(None)
                self.keys_per_site.append([])
        
        # Generate equal weights for all sites (1/num_sites)
        self.weights = torch.ones(self.num_sites, opt.n_class) / self.num_sites
    
    def __getitem__(self, index):
        # Determine which site and which sample to use
        site_idx = index % self.num_sites
        if len(self.keys_per_site[site_idx]) == 0:
            # If this site has no data, find the next site with data
            for i in range(self.num_sites):
                alt_site = (site_idx + i) % self.num_sites
                if len(self.keys_per_site[alt_site]) > 0:
                    site_idx = alt_site
                    break
        
        # Get a sample from this site
        if len(self.keys_per_site[site_idx]) == 0:
            raise RuntimeError("No data available in any site")
            
        key_idx = index // self.num_sites % len(self.keys_per_site[site_idx])
        key = self.keys_per_site[site_idx][key_idx]
        
        # Load image and label
        img_array = self.h5_files[site_idx][f'images/{key}'][()]
        label = self.h5_files[site_idx][f'labels/{key}'][()]
        img_tensor = self.transform(img_array)
        
        # Create the return dictionary in UAGAN-compatible format
        ret_dict = {}
        
        # For each site, we need to provide the data
        for i in range(self.num_sites):
            if i == site_idx:
                # For the selected site, use the actual data
                ret_dict[f'A_{i}'] = torch.tensor(label, dtype=torch.long)
                ret_dict[f'B_{i}'] = img_tensor
                ret_dict[f'A_paths_{i}'] = f"site_{i}/{key}"
                ret_dict[f'B_paths_{i}'] = f"site_{i}/{key}"
            else:
                # For other sites, provide empty data
                # (UAGAN will only use sites with valid weights)
                ret_dict[f'A_{i}'] = torch.tensor(label, dtype=torch.long)
                ret_dict[f'B_{i}'] = img_tensor  # Same image
                ret_dict[f'A_paths_{i}'] = f"placeholder"
                ret_dict[f'B_paths_{i}'] = f"placeholder"
        
        # Add weights tensor
        ret_dict['weights'] = self.weights
        
        return ret_dict
    
    def __len__(self):
        total_samples = 0
        for keys in self.keys_per_site:
            total_samples += len(keys)
        return total_samples