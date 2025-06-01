import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import random

def build_HAM10000_h5(data_dir, save_dir):
    """
    Convert HAM10000 dataset to H5 format.
    
    Parameters:
        data_dir: Directory containing HAM10000 data
        save_dir: Directory to save processed H5 files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load metadata
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    metadata = pd.read_csv(metadata_path)
    
    # Define image directory
    image_dir = os.path.join(data_dir, 'images')
    
    # Create train/test split (80/20)
    image_ids = list(metadata['image_id'])
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * 0.8)
    train_ids = image_ids[:split_idx]
    test_ids = image_ids[split_idx:]
    
    # Create mapping for diagnosis to numeric label
    label_map = {
        'akiec': 0,  # Actinic keratosis
        'bcc': 1,    # Basal cell carcinoma
        'bkl': 2,    # Benign keratosis
        'df': 3,     # Dermatofibroma
        'mel': 4,    # Melanoma
        'nv': 5,     # Melanocytic nevus
        'vasc': 6,   # Vascular lesion
    }
    
    # Create training file
    train_file = h5py.File(os.path.join(save_dir, 'train_HAM10000.h5'), 'w')
    print('Processing training files')
    for i, img_id in enumerate(tqdm(train_ids)):
        # Get image path
        img_path = os.path.join(image_dir, img_id + '.jpg')
        if not os.path.exists(img_path):
            # Try alternative extension
            img_path = os.path.join(image_dir, img_id + '.png')
        
        # Load and process image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((64, 64))  # Resize to consistent dimensions
        img_array = np.array(img)
        
        # Get label
        label = metadata[metadata['image_id'] == img_id]['dx'].values[0]
        numeric_label = label_map[label]
        
        # Store in H5 file
        train_file.create_dataset(f'images/{i}', data=img_array)
        train_file.create_dataset(f'labels/{i}', data=numeric_label)
    
    # Create test file
    test_file = h5py.File(os.path.join(save_dir, 'test_HAM10000.h5'), 'w')
    print('Processing test files')
    for i, img_id in enumerate(tqdm(test_ids)):
        # Get image path
        img_path = os.path.join(image_dir, img_id + '.jpg')
        if not os.path.exists(img_path):
            # Try alternative extension
            img_path = os.path.join(image_dir, img_id + '.png')
        
        # Load and process image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((64, 64))  # Resize to consistent dimensions
        img_array = np.array(img)
        
        # Get label
        label = metadata[metadata['image_id'] == img_id]['dx'].values[0]
        numeric_label = label_map[label]
        
        # Store in H5 file
        test_file.create_dataset(f'images/{i}', data=img_array)
        test_file.create_dataset(f'labels/{i}', data=numeric_label)
    
    print(f'Training: {len(train_file["images"])}')
    print(f'Test: {len(test_file["images"])}')
    train_file.close()
    test_file.close()

def build_HAM10000_split_by_site_h5(data_dir, save_dir):
    """
    Split HAM10000 dataset by acquisition site to simulate federated learning.
    
    Parameters:
        data_dir: Directory containing HAM10000 data
        save_dir: Directory to save processed H5 files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load metadata
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    metadata = pd.read_csv(metadata_path)
    
    # Define image directory
    image_dir = os.path.join(data_dir, 'images')
    
    # Create mapping for diagnosis to numeric label
    label_map = {
        'akiec': 0,  # Actinic keratosis
        'bcc': 1,    # Basal cell carcinoma
        'bkl': 2,    # Benign keratosis
        'df': 3,     # Dermatofibroma
        'mel': 4,    # Melanoma
        'nv': 5,     # Melanocytic nevus
        'vasc': 6,   # Vascular lesion
    }
    
    # Get unique acquisition sites instead of 'dataset'
    sites = metadata['acq_site'].unique()
    
    # Create a file for each site
    for site in sites:
        site_metadata = metadata[metadata['acq_site'] == site]
        site_file = h5py.File(os.path.join(save_dir, f'train_HAM10000_site_{site}.h5'), 'w')
        
        print(f'Processing site {site}')
        for i, (_, row) in enumerate(tqdm(site_metadata.iterrows())):
            img_id = row['image_id']
            
            # Get image path
            img_path = os.path.join(image_dir, img_id + '.jpg')
            if not os.path.exists(img_path):
                # Try alternative extension
                img_path = os.path.join(image_dir, img_id + '.png')
            
            # Load and process image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((64, 64))  # Resize to consistent dimensions
            img_array = np.array(img)
            
            # Get label
            label = row['dx']
            numeric_label = label_map[label]
            
            # Store in H5 file
            site_file.create_dataset(f'images/{i}', data=img_array)
            site_file.create_dataset(f'labels/{i}', data=numeric_label)
        
        print(f'Site {site}: {len(site_file["images"])} images')
        site_file.close()

if __name__ == "__main__":
    # Example usage:
    # build_HAM10000_h5('datasets/HAM10000', 'datasets/HAM10000_processed')
    # build_HAM10000_split_by_site_h5('datasets/HAM10000', 'datasets/HAM10000_processed')
    pass