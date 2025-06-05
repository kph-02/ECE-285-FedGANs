import h5py

def explore_h5(path):
    with h5py.File(path, 'r') as f:
        def visitor(name, obj):
            print(f"{name}: {type(obj)} shape={getattr(obj, 'shape', None)}")
        f.visititems(visitor)

explore_h5('/home/anojha/UAGAN/datasets/HAM10000_processed/train_HAM10000.h5')