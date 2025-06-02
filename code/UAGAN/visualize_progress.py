import os
import torch
import torchvision
from pathlib import Path
from options.test_options import TestOptions
from models import create_model

def generate_samples(opt, epoch=None, class_specific=True):
    """Generate visualization samples from a trained GAN model"""
    
    # Create and set up model
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    
    # Fix for device mismatch - ensure onehot is on the right device
    if hasattr(model, 'onehot'):
        model.onehot = model.onehot.to(model.device)
        print(f"Moved onehot tensor to device: {model.device}")
    
    # Create output directory
    if epoch is not None:
        sample_dir = Path(f"./samples/{opt.name}/epoch_{epoch}")
    else:
        sample_dir = Path(f"./samples/{opt.name}/latest")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        # Fixed noise vector for consistent samples
        batch_size = 10
        fixed_noise = torch.randn(batch_size, opt.nz, 1, 1, device=model.device)
        
        if class_specific:
            # Generate samples for each class
            for class_idx in range(7):  # HAM10000 has 7 classes
                # Create batch of identical class labels
                labels = torch.full((batch_size,), class_idx, dtype=torch.long, device=model.device)
                # Get one-hot encoding if model has it
                if hasattr(model, 'onehot'):
                    # Ensure both tensors are on the same device
                    onehot_labels = model.onehot[labels.to(model.onehot.device)].to(model.device)
                    fake_images = model.netG(fixed_noise, onehot_labels)
                else:
                    # Fallback if no onehot attribute
                    fake_images = model.netG(fixed_noise)
                
                # Save the images directly without make_grid
                img_path = sample_dir / f"class_{class_idx}.png"
                torchvision.utils.save_image(fake_images, img_path, normalize=True)
                print(f"Saved visualization for class {class_idx} to {img_path}")
        else:
            # Generate mixed samples
            # Random labels
            labels = torch.randint(0, 7, (batch_size,), device=model.device)
            # One-hot encode if available
            if hasattr(model, 'onehot'):
                # Fix device mismatch
                onehot_labels = model.onehot[labels.to(model.onehot.device)].to(model.device)
                fake_images = model.netG(fixed_noise, onehot_labels)
            else:
                fake_images = model.netG(fixed_noise)
                
            # Save the images directly
            img_path = sample_dir / "mixed_samples.png"
            torchvision.utils.save_image(fake_images, img_path, normalize=True)
            print(f"Saved mixed samples to {img_path}")

if __name__ == "__main__":
    """Generate samples from a trained model"""
    opt = TestOptions().parse()
    
    # Set required parameters if not provided
    if not hasattr(opt, 'nz'):
        opt.nz = 100  # Default noise dimension
        
    generate_samples(opt)
    print("Visualization complete!")