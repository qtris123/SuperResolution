"""
=============================================================================
SUPER-RESOLUTION MODEL EVALUATION
=============================================================================
This notebook demonstrates how to:
1. Load a trained super-resolution neural network
2. Test it on unseen data
3. Visualize the results with before/after comparisons
4. Calculate quantitative metrics (MSE and PSNR)

Super-resolution is the task of taking a low-resolution image and generating
a high-resolution version. Our model takes 225x300 images and outputs 450x600
images (2x upscaling).

A 2GB VRAM GPU is enough for the neural network
=============================================================================
"""


# Import necessary libraries for deep learning, image processing, and visualization
import torch  # Core PyTorch library for tensor operations and neural networks
import torch.nn as nn  # Neural network module from PyTorch
import torch.optim as optim  # Optimization algorithms (like Adam, SGD)
from torch.utils.data import Dataset, DataLoader, random_split  # Data utilities
from torchvision import transforms  # Image transformations
from PIL import Image  # Python Imaging Library for image handling
import os  # Operating system interface for file operations
import myutilities
from pathlib import Path
from torch.amp import autocast
from torch.amp import GradScaler
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import wandb 

from myutilities import lambda_lr, save_checkpoint



# Set device and enable BF16 if availsable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, otherwise CPU
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()  # Check if BF16 (bfloat16) is supported on the GPU
print(f"Using device: {device}")  # Print which device is being used
print(f"BF16 enabled: {use_bf16}")  # Print whether BF16 is enabled

# =============================================================================
# CUSTOM PIXEL SHUFFLE IMPLEMENTATION
# =============================================================================
class CustomPixelShuffle(nn.Module):
    """
    Hand-made implementation of PixelShuffle (also known as sub-pixel convolution).
    PyTorch has an implementation of it called nn.PixelShuffle
    
    This operation rearranges elements in a tensor from depth to spatial dimensions.
    It's a clever way to upscale images using learned convolutions instead of 
    simple interpolation.
    
    How it works:
    - Input:  (batch, channels * r^2, height, width)
    - Output: (batch, channels, height * r, width * r)
    
    where r is the upscale factor.
    
    Example with r=2:
    - Input:  (B, 12, 100, 100) where 12 = 3 * 2^2
    - Output: (B, 3, 200, 200)
    
    The magic is in the reshaping and permutation that rearranges the channel
    data into spatial locations.
    """
    def __init__(self, upscale_factor):
        super(CustomPixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        
    def forward(self, x):
        """
        Rearrange tensor to increase spatial resolution.
        
        Args:
            x: Input tensor of shape (batch, channels * r², height, width)
            
        Returns:
            Output tensor of shape (batch, channels, height * r, width * r)
        """
        r = self.upscale_factor
        
        # Get input dimensions
        b, c, h, w = x.shape
        
        # Step 1: Reshape to separate the upscale factor from channels
        # (b, c, h, w) -> (b, c/r², r, r, h, w)
        # This groups the channels that will become spatial pixels
        x = x.reshape(b, c // (r**2), r, r, h, w)
        
        # Step 2: Permute dimensions to interleave spatial information
        # (b, c/r², r, r, h, w) -> (b, c/r², h, r, w, r)
        # This moves the r×r blocks next to their spatial positions
        x = x.permute(0, 1, 4, 2, 5, 3)
        
        # Step 3: Collapse the r dimensions into spatial dimensions
        # (b, c/r², h, r, w, r) -> (b, c/r², h*r, w*r)
        # Now we have increased spatial resolution!
        x = x.reshape(b, c // (r**2), h * r, w * r)
        
        return x
      
# =============================================================================
# SUPER RESOLUTION CNN
# =============================================================================
   
class SuperResCNN(nn.Module):
    """
    A Convolutional Neural Network for 2x image super-resolution.
    
    Architecture:
    - Feature extraction: 3 convolutional layers to learn image patterns
    - Upsampling: Sub-pixel convolution (PixelShuffle) to increase resolution
    - Refinement: Final conv layer to polish the output
    
    Input:  3-channel RGB image of size 225x300
    Output: 3-channel RGB image of size 450x600 (2x larger)
    """
    def __init__(self):
        super(SuperResCNN, self).__init__()
        # Feature extraction layers - learn to recognize patterns like edges and textures
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4, device="cuda")   # First conv layer
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2, device="cuda")   # Second conv layer
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, padding=2, device="cuda")   # Third conv layer
        
        # Upsampling branch - increases image resolution by 2x
        self.upsample_conv = nn.Conv2d(32, 12, kernel_size=3, padding=1, device="cuda")
        # We could use PyTorch's implementation of PixelShuffle
        # self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)      # Rearranges channels to spatial dimensions
        # But we will be using our own implementation of PixelShuffle
        self.pixel_shuffle = CustomPixelShuffle(upscale_factor=2)
        
        # Final refinement layer - smooths out artifacts
        self.refine = nn.Conv2d(3, 3, kernel_size=3, padding=1, device="cuda")

    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch_size, 3, 225, 300)
        Returns:
            Output tensor of shape (batch_size, 3, 450, 600)
        """
        identity = F.interpolate(x, scale_factor = 2, mode="bilinear", align_corners=False)
        # Feature extraction with ReLU activation
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Upsampling to increase resolution
        x = self.upsample_conv(x)
        x = self.pixel_shuffle(x)
        
        # Final refinement
        x = self.refine(x)
        return x + identity


    
# ============================================================
# Train & Validate
# ============================================================
@torch.no_grad()
def calc_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    """
    pred, target: [B,3,H,W] in [0,1]
    """
    mse = torch.mean((pred.clamp(0,1) - target.clamp(0,1)) ** 2).item()
    if mse <= eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)

def train_k_epoch(save_dir, eval_step, epoch, model, train_loader, eval_loader, criterion, optimizer, scheduler, scaler, clip_grad: float = 1.0):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    steps = 0
    best_psnr = 0
    best_epoch = "Null"

    plot_loss = []
    plot_psnr = []
    plot_eval_loss = []
    plot_eval_psnr = []
    for k in range(epoch):
        local_loss, local_psnr = 0, 0
        for lr, hr in train_loader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            #with autocast(device_type="cuda"):
            sr = model.forward(lr)
            loss = criterion(sr, hr)

            scaler.scale(loss).backward()
            if clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                psnr = calc_psnr(sr, hr)
                if psnr > best_psnr:
                    #best_epoch = k
                    best_psnr = psnr

            local_loss += loss.item()
            local_psnr += psnr
            total_loss += loss.item()
            total_psnr += psnr
            plot_loss.append(loss.item())
            plot_psnr.append(psnr)
            wandb.log({"train/loss": loss.item(),
                       "train/psnr": psnr},)
            steps += 1
            print(f"At step {k}: Local loss is {local_loss} and Local psnr is {local_psnr}")
            
            if steps % eval_step == 0:
                eval_loss, eval_psnr = validate(model, eval_loader, criterion)

                plot_eval_loss.append(eval_loss)
                plot_eval_psnr.append(eval_psnr)
                wandb.log({"eval/loss": eval_loss,
                           "eval/psnr": eval_psnr})
        
    save_checkpoint(model, optimizer, scaler, k, best_psnr, best_epoch, save_dir)

    return model, best_epoch, total_loss / (steps*epoch), total_psnr / (steps*epoch), plot_loss, plot_psnr, plot_eval_loss, plot_eval_psnr

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    steps = 0

    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        #with autocast(device_type="cuda"):
        sr = model.forward(lr)
        loss = criterion(sr, hr)

        psnr = calc_psnr(sr, hr)
        total_loss += loss.item()
        total_psnr += psnr
        steps += 1

    return total_loss / steps, total_psnr / steps

def main():
    # Load and prepare the dataset
    print("Loading dataset...")
    dataset = myutilities.ImageSuperResDataset('kaggle/train_x', 'kaggle/train_y')
    save_dir = "checkpoints/"
    os.makedirs(save_dir, exist_ok=True)
    eval_step = 50

    # Split dataset into training, validation, and test sets (70%/15%/15%)
    total_size = len(dataset)
    train_size = int(0.70 * total_size)  # 70% for training
    val_size = int(0.15 * total_size)    # 15% for validation
    test_size = total_size - train_size - val_size  # Remaining 15% for testing

    # Use random_split to create subsets and set a seed for reproducibility
    # We do not need the train and validation data now
    train_dataset, eval_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for consistent splits
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create data loaders for efficient batch processing
    lr = 0.0005
    epoch = 3
    batch_size = 10  # Number of images to process in each batch (adjust based on GPU memory)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # =============================================================================
    # SETUP: Device Configuration
    # =============================================================================
    # Check if GPU is available - neural networks run much faster on GPUs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SuperResCNN()
    ## Negative log-likelihood loss
    ## This is a binary classification problem
    criterion = nn.L1Loss()
    ## Stochastic Gradient Descent
    #### Just like gradient descent, but uses just part of the data not the entire dataset
    optimizer = optim.AdamW(model.parameters(), lr = lr, betas = (0.9, 0.999), weight_decay=0.01)
    scheduler = LambdaLR(optimizer, lambda_lr)
    scaler = GradScaler()
    
    #Wandb
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="vqtri-purdue-university",
        # Set the wandb project where this run will be logged.
        project="SuperResolution",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": lr,
            "architecture": "3 convolutions",
            "epochs": epoch,
            "batch_size": batch_size,
            "eval_step": eval_step,
            "scheduler": "constant with warm up 100 steps"
        },
    )
    #Train
    model, best_epoch, avg_loss, avg_psnr, plot_loss, plot_psnr, plot_eval_loss, plot_eval_pnsrn = train_k_epoch(save_dir, eval_step, epoch, model, train_loader, eval_loader, criterion, optimizer, scheduler, scaler)
    myutilities.plot_loss_psnr("Train", plot_loss, "L1_Loss", plot_psnr, "PSNR")
    myutilities.plot_loss_psnr("Eval", plot_eval_loss, "Eval L1 loss", plot_eval_pnsrn, "Eval PSNR")
    print(f"Avg loss: {avg_loss}")
    print(f"Avg psnr: {avg_psnr}")

if __name__ == "__main__":
    main()
    
def load_checkpoint(path="checkpoint.pth", device="cuda"):
    checkpoint = torch.load(path, map_location=device)

    model = SuperResCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])

    optimizer = torch.optim.Adam(model.parameters())  # placeholder
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    scaler = GradScaler()
    if checkpoint["scaler_state"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])

    start_epoch = checkpoint["epoch"] + 1
    best_psnr = checkpoint.get("best_psnr", None)

    print(f"Loaded checkpoint from {path}, resuming at epoch {start_epoch}")
    return model, optimizer, scaler, start_epoch, best_psnr
    