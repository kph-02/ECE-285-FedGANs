import pandas as pd
import os

# Adjust this path to match your dataset location
data_dir = './datasets/HAM10000'
metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')

# Load the metadata
metadata = pd.read_csv(metadata_path)

# Count unique lesion_ids
total_lesion_count = metadata['lesion_id'].nunique()
print(f"Total unique lesion IDs: {total_lesion_count}")

# Extract prefixes (potential site indicators)
metadata['prefix'] = metadata['lesion_id'].apply(lambda x: x.split('_')[0])
prefix_counts = metadata['prefix'].value_counts()

print("\nPrefix counts (possible sites):")
print(prefix_counts)

# Count by diagnosis
localization_counts = metadata['localization'].value_counts()
print("\nlocalization counts:")
print(localization_counts)
