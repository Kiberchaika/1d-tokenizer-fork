import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import torchvision.utils as vutils

from modeling.titok import TiTok
from modeling.quantizer import VectorQuantizer
from demo_util import get_config

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

config = get_config("configs/titok_mnist.yaml")
model = TiTok(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Initialize TensorBoard writer
writer = SummaryWriter('runs/titok_mnist_experiment')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)

import os
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
os.makedirs("checkpoints", exist_ok=True)

optimizer = optim.Adam(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

beta = 0.1  # Adjust this value
global_step = 0

for epoch in range(10):
    for i, (data, _) in enumerate(train_loader):
        data = data.to(device)
        
        with autocast():
            z_quantized, result_dict = model.encode(data)
            reconstructed = model.decode(z_quantized)
            loss = beta * result_dict['quantizer_loss'] + F.mse_loss(reconstructed, data)
        
        scaler.scale(loss).backward()
        
        if (i + 1) % config.training.accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Logging
        if i % 100 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Quantizer_Loss/train', result_dict['quantizer_loss'].item(), global_step)
            
            original_grid = vutils.make_grid(data[:8].cpu(), normalize=True, scale_each=True)
            reconstructed_grid = vutils.make_grid(reconstructed[:8].cpu(), normalize=True, scale_each=True)
            writer.add_image('Original', original_grid, global_step)
            writer.add_image('Reconstructed', reconstructed_grid, global_step)

        global_step += 1

    scheduler.step()
    torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pth")

writer.close()