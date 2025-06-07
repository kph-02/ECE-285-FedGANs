#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import argparse

def add_acq_site_column(csv_path, num_sites=10, seed=42):
    """
    Add an acquisition site column to HAM10000_metadata.csv
    
    Args:
        csv_path: Path to the HAM10000_metadata.csv file
        num_sites: Number of acquisition sites to create
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Add acq_site column with random values from 0 to num_sites-1
    df['acq_site'] = np.random.randint(0, num_sites, size=len(df))
    
    # Save the updated CSV
    df.to_csv(csv_path, index=False)
    
    print(f"Added 'acq_site' column with {num_sites} possible values to {csv_path}")
    
    # Return distribution of sites for verification
    site_dist = df['acq_site'].value_counts().sort_index()
    return site_dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add acquisition site column to HAM10000 dataset')
    parser.add_argument('--num_sites', type=int, default=10, 
                        help='Number of acquisition sites to create')
    parser.add_argument('--csv_path', type=str, default='HAM10000_metadata.csv',
                        help='Path to the HAM10000_metadata.csv file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.isfile(args.csv_path):
        print(f"Error: File '{args.csv_path}' not found!")
    else:
        # Add the acq_site column
        distribution = add_acq_site_column(args.csv_path, args.num_sites, args.seed)
        print("\nDistribution of acquisition sites:")
        for site, count in distribution.items():
            print(f"Site {site}: {count} samples ({count/len(pd.read_csv(args.csv_path))*100:.1f}%)")