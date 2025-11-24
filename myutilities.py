import torch  # Core PyTorch library for tensor operations and neural networks
import torch.nn as nn  # Neural network module from PyTorch
import torch.optim as optim  # Optimization algorithms (like Adam, SGD)
from torch.utils.data import Dataset, DataLoader, random_split  # Data utilities
from torchvision import transforms  # Image transformations
from PIL import Image  # Python Imaging Library for image handling
import os  # Operating system interface for file operations
import matplotlib.pyplot as plt  # Plotting library for visualizing results
import numpy as np
from torch.amp import GradScaler

# =============================================================================
# Custom Dataset Class for Image Super-Resolution
# =============================================================================

class ImageSuperResDataset(Dataset):
    def __init__(self, input_dir, output_dir):
        """
        Initialize the dataset with directories containing low-res and high-res images.
        
        Args:
            input_dir: Directory containing low-resolution input images (225x300)
            output_dir: Directory corresponding high-resolution target images (450x600)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        # Get sorted list of filenames to ensure consistent ordering
        self.filenames = sorted(os.listdir(input_dir))
        
        # Define transformations for input and output images
        self.input_transform = transforms.Compose([
            transforms.Resize((225, 300)),  # Resize all inputs to standard size (225x300)
            transforms.ToTensor()  # Convert PIL image to PyTorch tensor
        ])
        self.output_transform = transforms.Compose([
            transforms.Resize((450, 600)),  # Resize all outputs to double the input dimensions (450x600)
            transforms.ToTensor()  # Convert PIL image to PyTorch tensor
        ])
    
    def __len__(self):
        """Return the total number of images in the dataset"""
        return len(self.filenames)
    
    def __getitem__(self, idx):
        """
        Retrieve a pair of low-res and high-res images by index.
        
        Args:
            idx: Index of the image to retrieve
            
        Returns:
            A tuple containing (input_tensor, output_tensor) where input is 3x225x300
            and output is 3x450x600 (channels x height x width)
        """
        img_name = self.filenames[idx]
        
        # Load low-resolution input image
        input_path = os.path.join(self.input_dir, img_name)
        input_img = Image.open(input_path).convert('RGB')  # Ensure RGB format
        input_tensor = self.input_transform(input_img)
        
        # Load high-resolution target image
        output_path = os.path.join(self.output_dir, img_name)
        output_img = Image.open(output_path).convert('RGB')  # Ensure RGB format
        output_tensor = self.output_transform(output_img)
        
        # Return images as 2D tensors (C x H x W) without flattening
        return input_tensor, output_tensor
    

# =============================================================================
# VISUALIZATION FUNCTION
# =============================================================================
def visualize_super_resolution(model, test_loader, device, num_examples=5):
    """
    Visualize super-resolution results with side-by-side comparisons.
    
    For each example, we show:
    1. Input: Low-resolution image (225x300)
    2. Model Output: Super-resolved image (450x600)
    3. Ground Truth: Actual high-resolution image (450x600)
    
    Args:
        model: The trained super-resolution model
        test_loader: DataLoader containing test images
        num_examples: Number of examples to visualize (default: 3)
    """
    model.eval()
    data_iter = iter(test_loader)
    inputs, targets = next(data_iter)
    
    # Limit to requested number of examples and move to device
    inputs = inputs[:num_examples].to(device)
    targets = targets[:num_examples].to(device)
    
    print(f"\nGenerating super-resolution for {num_examples} test images...")
    
    # Generate predictions (no gradient computation needed for inference)
    with torch.no_grad():
        outputs = model(inputs)
    
    # Move tensors to CPU and convert to numpy arrays for plotting
    inputs_np = inputs.cpu().numpy()
    outputs_np = outputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Create a grid of subplots: one row per example, three columns per row
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5 * num_examples))
    
    # Handle case of single example (axes would be 1D)
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    # Process each example
    for i in range(num_examples):
        # Convert from PyTorch format (C, H, W) to matplotlib format (H, W, C)
        input_img = np.transpose(inputs_np[i], (1, 2, 0))
        output_img = np.transpose(outputs_np[i], (1, 2, 0))
        target_img = np.transpose(targets_np[i], (1, 2, 0))
        
        # Ensure pixel values are in valid range [0, 1]
        input_img = np.clip(input_img, 0, 1)
        output_img = np.clip(output_img, 0, 1)
        target_img = np.clip(target_img, 0, 1)
        
        # Display the three images side by side
        axes[i, 0].imshow(input_img)
        axes[i, 0].set_title(f'Input (Low-Res)\n{inputs_np[i].shape[1]}x{inputs_np[i].shape[2]}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(output_img)
        axes[i, 1].set_title(f'Model Output\n{outputs_np[i].shape[1]}x{outputs_np[i].shape[2]}')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(target_img)
        axes[i, 2].set_title(f'Ground Truth\n{targets_np[i].shape[1]}x{targets_np[i].shape[2]}')
        axes[i, 2].axis('off')
        
        # Calculate quality metrics
        mse = np.mean((output_img - target_img) ** 2)
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))  # Add small epsilon to avoid division by zero
        
        print(f"  Example {i+1} - MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")
    
    plt.tight_layout()
    #plt.savefig('superres_results.png', dpi=150, bbox_inches='tight')
    #print(f"\nVisualization saved as 'superres_results.png'")
    plt.show()

# =============================================================================
# QUANTITATIVE EVALUATION FUNCTION
# =============================================================================
def evaluate_metrics(model, test_loader, device):
    """
    Calculate average quality metrics across the entire test set.
    
    Metrics:
    - MSE (Mean Squared Error): Lower is better, measures pixel-wise difference
    - PSNR (Peak Signal-to-Noise Ratio): Higher is better, logarithmic quality measure
    
    Args:
        model: The trained super-resolution model
        test_loader: DataLoader containing all test images
        
    Returns:
        avg_mse: Average mean squared error
        avg_psnr: Average PSNR in decibels
    """
    model.eval()
    
    # Accumulate metrics across all batches
    total_mse = 0
    total_psnr = 0
    count = 0
    
    print("\nEvaluating model on test set...")
    
    # Process all test images
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Generate predictions
            outputs = model(inputs)
            
            # Calculate MSE and PSNR for each image in the batch
            mse = torch.mean((outputs - targets) ** 2, dim=[1, 2, 3])
            psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
            
            # Accumulate statistics
            total_mse += mse.sum().item()
            total_psnr += psnr.sum().item()
            count += inputs.size(0)
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {count} images...")
    
    # Calculate averages
    avg_mse = total_mse / count
    avg_psnr = total_psnr / count
    
    # Print results
    print(f"\n{'='*70}")
    print(f"TEST SET EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Total images evaluated: {count}")
    print(f"Average MSE:  {avg_mse:.6f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    # Simple quality assessment
    if avg_psnr >= 30:
        quality = "Good"
    else:
        quality = "Better hyperparameter search needed"
        
    print(f"Quality Assessment: {quality}")
    print(f"{'='*70}\n")
    
    return avg_mse, avg_psnr
print("check")

def plot_loss_metrics(name, plot_loss, loss_name, plot_metric, metric_name):
    x1 = range(len(plot_loss))
    x2 = range(len(plot_metric))

    plt.figure(figsize=(10, 5))

    # --- Plot 1 ---
    plt.subplot(1, 2, 1)
    plt.plot(x1, plot_loss, linewidth=2)
    plt.title(f"Plot {loss_name}")
    plt.xlabel("Index")
    plt.ylabel("Value")

    # --- Plot 2 ---
    plt.subplot(1, 2, 2)
    plt.plot(x2, plot_metric, linewidth=2)
    plt.title(f"Plot {metric_name}")
    plt.xlabel("Index")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.savefig(f"{name}_two_plots.png", dpi=300)
    plt.close()
    
    
def lambda_lr(cur_step):
    # constant with warm up 100 steps
    warmup_steps = 100
    if cur_step < 100:
        return float(cur_step / float(warmup_steps))
    return 1.0

def save_checkpoint(model, optimizer, scaler, epoch, best_psnr, best_checkpoint, path="checkpoint.pth"):
    path = path + f"epoch_{best_checkpoint}"
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_psnr": best_psnr,
        "checkpoint": best_checkpoint,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")
