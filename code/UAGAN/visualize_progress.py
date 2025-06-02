import os
import torch
import torchvision
from pathlib import Path
from options.test_options import TestOptions
from models import create_model
import matplotlib.pyplot as plt

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

def generate_class_samples(opt, num_samples=4):
    """Generate and visualize samples from a trained GAN model, 4 samples per class"""
    
    # Class names for HAM10000
    HAM_CLASSES = {
        0: "Actinic keratoses",
        1: "Basal cell carcinoma",
        2: "Benign keratosis",
        3: "Dermatofibroma",
        4: "Melanoma",
        5: "Melanocytic nevi",
        6: "Vascular lesions"
    }
    
    # Determine number of classes based on model type
    num_classes = opt.n_class  # Use what's specified in options
    
    # Create and setup model
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    
    # Fix for device mismatch - ensure onehot is on the right device
    if hasattr(model, 'onehot'):
        model.onehot = model.onehot.to(model.device)
        print(f"Moved onehot tensor to device: {model.device}")
    
    # Create output directory
    sample_dir = Path(f"./samples/{opt.name}/{opt.epoch}")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots for all classes
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(num_samples*3, num_classes*3))
    fig.suptitle(f"Generated Samples from {opt.name} - {opt.model} model", fontsize=16)
    
    with torch.no_grad():
        # Samples per class
        for class_idx in range(num_classes):
            if class_idx >= 7:  # Skip non-existent classes (in case n_class > 7)
                continue
                
            # Get class name for display
            class_name = HAM_CLASSES.get(class_idx, f"Class {class_idx}")
            
            for sample_idx in range(num_samples):
                # Generate new noise for each sample
                noise = torch.randn(1, opt.nz, 1, 1, device=model.device)
                
                # Create label
                label = torch.tensor([class_idx], device=model.device)
                
                # Generate image
                if hasattr(model, 'onehot'):
                    # One-hot encoding for conditional GAN
                    onehot_label = model.onehot[label].to(model.device)
                    fake_image = model.netG(noise, onehot_label)
                else:
                    # For non-conditional GAN
                    fake_image = model.netG(noise)
                
                # Convert to numpy for plotting
                img_np = fake_image[0].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np + 1) / 2.0  # Normalize from [-1,1] to [0,1]
                
                # Plot in the grid
                ax = axes[class_idx, sample_idx]
                ax.imshow(img_np)
                if sample_idx == 0:  # Only on the first column
                    ax.set_ylabel(class_name, fontsize=11)
                ax.axis('off')
                
            # Save individual class row as separate image too
            class_fig, class_axes = plt.subplots(1, num_samples, figsize=(num_samples*3, 3))
            for i in range(num_samples):
                # Get the corresponding image from the main figure
                img = axes[class_idx, i].images[0].get_array()
                class_axes[i].imshow(img)
                class_axes[i].axis('off')
            
            class_fig.suptitle(f"{class_name} (Class {class_idx})", fontsize=14)
            class_fig.tight_layout()
            class_path = sample_dir / f"class_{class_idx}_{class_name}.png"
            class_fig.savefig(class_path)
            plt.close(class_fig)
            print(f"Saved class {class_idx} ({class_name}) to {class_path}")
    
    # Save the combined figure
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)  # Adjust for the title
    combined_path = sample_dir / "all_classes.png"
    fig.savefig(combined_path)
    print(f"Saved combined visualization to {combined_path}")
    
    # Display the figure if running in an interactive environment
    plt.show()
if __name__ == "__main__":
    """Generate samples from a trained model"""
    opt = TestOptions().parse()
    
    # Set required parameters if not provided
    if not hasattr(opt, 'nz'):
        opt.nz = 100  # Default noise dimension
        
    generate_class_samples(opt)
    print("Visualization complete!")