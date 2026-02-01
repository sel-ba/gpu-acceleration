import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.profiler import profile, record_function, ProfilerActivity
import time
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import urllib.request
import tarfile
import os


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(UNet, self).__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.bottleneck = conv_block(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        self.out = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)


class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.img_dir = self.root_dir / 'images'
        self.mask_dir = self.root_dir / 'annotations' / 'trimaps'
        
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        
        self.images = self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        n_samples = 500 if self.split == 'train' else 100
        images = []
        
        for i in range(n_samples):
            img_path = self.img_dir / f'{self.split}_{i}.jpg'
            mask_path = self.mask_dir / f'{self.split}_{i}.png'
            
            if not img_path.exists():
                img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                Image.fromarray(img).save(img_path)
                
                mask = np.random.randint(0, 3, (128, 128), dtype=np.uint8)
                Image.fromarray(mask).save(mask_path)
            
            images.append((str(img_path), str(mask_path)))
        
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.images[idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask).long().squeeze(0)
        
        return image, mask


class UNetTrainer:
    def __init__(self, device, batch_size=16, num_workers=4):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.setup_data()
        self.setup_model()
        
    def setup_data(self):
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        trainset = OxfordPetDataset('./data/oxford_pets', split='train', transform=transform)
        testset = OxfordPetDataset('./data/oxford_pets', split='test', transform=transform)
        
        self.trainloader = DataLoader(
            trainset, batch_size=self.batch_size, 
            shuffle=True, num_workers=0, pin_memory=False
        )
        self.testloader = DataLoader(
            testset, batch_size=self.batch_size,
            shuffle=False, num_workers=0, pin_memory=False
        )
    
    def setup_model(self):
        self.model = UNet(in_channels=3, num_classes=3)
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2)
    
    def dice_coefficient(self, pred, target):
        smooth = 1.0
        pred = torch.argmax(pred, dim=1)
        intersection = (pred == target).float().sum()
        return (2. * intersection + smooth) / (pred.numel() + target.numel() + smooth)
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_dice = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            dice = self.dice_coefficient(outputs, targets)
            running_dice += dice.item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch: {epoch} [{batch_idx}/{len(self.trainloader)}] '
                      f'Loss: {running_loss/(batch_idx+1):.3f} '
                      f'Dice: {running_dice/(batch_idx+1):.3f}')
        
        return running_loss / len(self.trainloader), running_dice / len(self.trainloader)
    
    def evaluate(self):
        self.model.eval()
        test_loss = 0
        test_dice = 0
        
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                dice = self.dice_coefficient(outputs, targets)
                test_dice += dice.item()
        
        return test_loss / len(self.testloader), test_dice / len(self.testloader)
    
    def train_with_profiling(self, epochs, profile_enabled=True):
        results = {
            'device': self.device,
            'epochs': epochs,
            'batch_size': self.batch_size,
            'train_losses': [],
            'train_dice': [],
            'test_losses': [],
            'test_dice': [],
            'epoch_times': [],
            'total_time': 0,
            'profiler_path': None
        }
        
        start_time = time.time()
        
        if profile_enabled and self.device == 'cuda':
            torch.cuda.synchronize()
        
        profiler_dir = Path(f'runs/unet_{self.device}')
        profiler_dir.mkdir(parents=True, exist_ok=True)
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if self.device == 'cuda' else [ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profiler_dir))
        ) as prof:
            for epoch in range(epochs):
                epoch_start = time.time()
                
                with record_function("train_epoch"):
                    train_loss, train_dice = self.train_epoch(epoch)
                
                with record_function("evaluate"):
                    test_loss, test_dice = self.evaluate()
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                epoch_time = time.time() - epoch_start
                
                results['train_losses'].append(train_loss)
                results['train_dice'].append(train_dice)
                results['test_losses'].append(test_loss)
                results['test_dice'].append(test_dice)
                results['epoch_times'].append(epoch_time)
                
                self.scheduler.step(test_loss)
                
                print(f'Epoch {epoch}: Train Loss: {train_loss:.3f}, Train Dice: {train_dice:.3f}, '
                      f'Test Loss: {test_loss:.3f}, Test Dice: {test_dice:.3f}, Time: {epoch_time:.2f}s')
                
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
                prof.step()
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        results['total_time'] = total_time
        results['profiler_path'] = str(profiler_dir)
        
        print(f'\nTotal training time: {total_time:.2f}s')
        print(f'Average time per epoch: {total_time/epochs:.2f}s')
        
        results_path = profiler_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'Results saved to {results_path}')
        
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"Training U-Net on {args.device.upper()}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    
    trainer = UNetTrainer(args.device, args.batch_size, args.num_workers)
    results = trainer.train_with_profiling(args.epochs)
    
    print(f"\nTraining completed on {args.device.upper()}")
    print(f"Final test Dice coefficient: {results['test_dice'][-1]:.3f}")
    
    if args.device == 'cuda':
        torch.cuda.empty_cache()
        print("GPU memory cleared")


if __name__ == '__main__':
    main()
